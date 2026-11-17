"""Central GPU inference server for batched self-play / evaluation.

Search workers run MCTS on the CPU but do not evaluate the network
themselves: each sends the encoded leaf positions of a search round (for
both agents at once) to one server process that owns the GPU. The server
waits until every worker's request is in, merges them into one batch per
model and runs everything else of the evaluation on the GPU as well: the
random board symmetry of each position, the network, mapping the priors
back to the original orientation and normalising them.

Protocol (multiprocessing queues):
    request_q  <- ('infer', worker_id, [(model_key, uint8 planes,
                                         symmetry indices), ...])
               <- ('done', worker_id)
    response_qs[worker_id] <- [(priors, values), ...]   (one per part)
"""
import queue as _queue
import time

import numpy as np

__all__ = [
    'RemoteInference',
    'run_inference_server',
]


class RemoteInference:
    """A worker's connection to the inference server for one model."""

    def __init__(self, request_q, response_q, worker_id, model_key,
                 timeout=600):
        self.request_q = request_q
        self.response_q = response_q
        self.worker_id = worker_id
        self.model_key = model_key
        self.timeout = timeout

    def same_connection(self, other):
        return (other.worker_id == self.worker_id and
                other.response_q is self.response_q)

    def evaluate(self, parts):
        """parts: [(model_key, uint8 planes (n, planes, N, N), symmetry
        indices (n,))]. Returns [(priors (n, N*N+1), values (n,))], priors
        normalised and in the original orientation."""
        self.request_q.put((
            'infer', self.worker_id,
            [(key, np.asarray(planes, dtype=np.uint8),
              np.asarray(indices, dtype=np.int32))
             for key, planes, indices in parts]))
        return self.response_q.get(timeout=self.timeout)


def symmetry_tables(board_size):
    """Index tables for the 8 board symmetries (see agent._sym):
    forward[s, j] = flat point that symmetry s moves to point j;
    inverse[s] maps a move vector (with pass last) back from symmetry s."""
    from .agent import _INVERSE, _sym
    n2 = board_size * board_size
    base = np.arange(n2).reshape(board_size, board_size)
    forward = np.stack([_sym(base, s).flatten() for s in range(8)])
    inverse = np.stack([
        np.concatenate([forward[_INVERSE[s]], [n2]]) for s in range(8)])
    return forward.astype(np.int32), inverse.astype(np.int32)


def make_evaluator(model):
    """tf.function: (uint8 planes, symmetry indices) -> (priors, values),
    doing the symmetry transforms on the GPU."""
    import tensorflow as tf
    _, planes, board_size, _ = model.input_shape
    n2 = board_size * board_size
    forward, inverse = symmetry_tables(board_size)
    forward = tf.constant(forward)
    inverse = tf.constant(inverse)

    @tf.function(input_signature=[
        tf.TensorSpec([None, planes, board_size, board_size], tf.uint8),
        tf.TensorSpec([None], tf.int32)])
    def evaluate(x, indices):
        n = tf.shape(x)[0]
        flat = tf.reshape(tf.cast(x, tf.float32), [n, planes, n2])
        flat = tf.gather(flat, tf.gather(forward, indices), axis=2,
                         batch_dims=1)
        priors, values = model(
            tf.reshape(flat, [n, planes, board_size, board_size]),
            training=False)
        priors = tf.gather(priors, tf.gather(inverse, indices), axis=1,
                           batch_dims=1)
        priors = priors / tf.reduce_sum(priors, axis=1, keepdims=True)
        return priors, values[:, 0]

    return evaluate


def run_inference_server(model_files, request_q, response_qs,
                         max_wait=0.05, label='[inference server] '):
    """Serve network evaluations until every worker has sent 'done'.

    model_files[k] is the agent file for model_key k. Each active worker
    has at most one request outstanding; the server collects until all
    active workers have sent theirs (or `max_wait` seconds after the first
    one), so every network call gets the positions of all workers.
    """
    import h5py
    import tensorflow as tf
    from dlgo import kerasutil

    kerasutil.set_gpu_memory_target(1.0)
    evaluators = []
    for filename in model_files:
        with h5py.File(filename, 'r') as h5file:
            model = kerasutil.load_model_from_hdf5_group(h5file['model'])
        evaluators.append(make_evaluator(model))

    active = set(range(len(response_qs)))
    calls = positions = 0
    busy = 0.0
    start = time.time()
    while active:
        pending = []

        def take(msg):
            if msg[0] == 'done':
                active.discard(msg[1])
            else:
                pending.append(msg)

        take(request_q.get())
        deadline = time.perf_counter() + max_wait
        while len(pending) < len(active):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                take(request_q.get(timeout=remaining))
            except _queue.Empty:
                break
        if not pending:
            continue

        t0 = time.perf_counter()
        replies = {msg[1]: [None] * len(msg[2]) for msg in pending}
        keys = sorted({part[0] for msg in pending for part in msg[2]})
        for key in keys:
            where = [(msg[1], i, part) for msg in pending
                     for i, part in enumerate(msg[2]) if part[0] == key]
            planes = np.concatenate([part[1] for _, _, part in where])
            indices = np.concatenate([part[2] for _, _, part in where])
            priors, values = evaluators[key](tf.constant(planes),
                                             tf.constant(indices))
            priors, values = priors.numpy(), values.numpy()
            calls += 1
            positions += len(planes)
            offset = 0
            for worker_id, i, part in where:
                n = len(part[1])
                replies[worker_id][i] = (priors[offset:offset + n],
                                         values[offset:offset + n])
                offset += n
        for worker_id, reply in replies.items():
            response_qs[worker_id].put(reply)
        busy += time.perf_counter() - t0

    elapsed = time.time() - start
    if calls:
        print('%s%d network calls, %.1f positions per call, network busy '
              '%.0f%% of %.0f s' % (label, calls, positions / calls,
                                    100.0 * busy / max(elapsed, 1e-9),
                                    elapsed))
