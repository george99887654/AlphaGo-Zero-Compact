"""Central GPU inference server for batched self-play / evaluation.

Search workers run MCTS on the CPU but do not evaluate the network
themselves: each sends its batch of leaf positions to one server process
that owns the GPU, which merges the requests of all workers into a single
network call per model and sends every worker its share of the results.
This moves all network work off the CPU and gives the GPU large batches.

Protocol (multiprocessing queues):
    request_q  <- ('infer', worker_id, model_key, float32 array)
               <- ('done', worker_id)
    response_qs[worker_id] <- (priors, values)
"""
import queue as _queue
import time

import numpy as np

__all__ = [
    'RemoteInference',
    'run_inference_server',
]


class RemoteInference:
    """Drop-in replacement for ZeroAgent.infer that asks the server."""

    def __init__(self, request_q, response_q, worker_id, model_key,
                 timeout=600):
        self.request_q = request_q
        self.response_q = response_q
        self.worker_id = worker_id
        self.model_key = model_key
        self.timeout = timeout

    def __call__(self, model_input):
        self.request_q.put(('infer', self.worker_id, self.model_key,
                            np.asarray(model_input, dtype=np.float32)))
        return self.response_q.get(timeout=self.timeout)


def run_inference_server(model_files, request_q, response_qs,
                         max_wait=0.003, label='[inference server] '):
    """Serve network evaluations until every worker has sent 'done'.

    model_files[k] is the agent file for model_key k. After the first
    request arrives the server waits up to `max_wait` seconds for the other
    active workers' requests, so batches fill up without stalling anyone.
    """
    import h5py
    import tensorflow as tf
    from dlgo import kerasutil

    kerasutil.set_gpu_memory_target(1.0)
    fns = []
    for filename in model_files:
        with h5py.File(filename, 'r') as h5file:
            model = kerasutil.load_model_from_hdf5_group(h5file['model'])
        fns.append(tf.function(lambda x, m=model: m(x, training=False),
                               reduce_retracing=True))

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
        # Every active worker has at most one request outstanding.
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
        for key in sorted({msg[2] for msg in pending}):
            group = [msg for msg in pending if msg[2] == key]
            batch = np.concatenate([msg[3] for msg in group])
            priors, values = fns[key](tf.convert_to_tensor(batch))
            priors, values = priors.numpy(), values.numpy()
            calls += 1
            positions += len(batch)
            offset = 0
            for _, worker_id, _, arr in group:
                n = len(arr)
                response_qs[worker_id].put(
                    (priors[offset:offset + n], values[offset:offset + n]))
                offset += n
        busy += time.perf_counter() - t0

    elapsed = time.time() - start
    if calls:
        print('%s%d network calls, %.1f positions per call, network busy '
              '%.0f%% of %.0f s' % (label, calls, positions / calls,
                                    100.0 * busy / max(elapsed, 1e-9),
                                    elapsed))
