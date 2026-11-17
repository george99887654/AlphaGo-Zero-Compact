import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from .. import kerasutil
from .. import goboard_fast
from .. import scoring
from .. import encoders

from dlgo.agent.helpers_fast import is_point_an_eye 

from ..agent import Agent

__all__ = [
    'ZeroAgent',
    'evaluate_many',
    'load_zero_agent',
    'select_moves_multi',
]

def legal_non_eye_moves(state, all_moves):
    """Indices of all_moves that are legal in `state` and do not fill one of
    the mover's own eyes.

    Same result as
        [i for i, m in enumerate(all_moves)
         if state.is_valid_move(m)
         and not is_point_an_eye(state.board, m.point, state.next_player)]
    but checks "game over" once and calls the board directly, since this
    runs for every new search node.
    """
    if state.is_over():
        return []
    board = state.board
    grid = board._grid
    neighbor_table = board.neighbor_table
    corner_table = board.corner_table
    player = state.next_player
    legal = []
    for idx, move in enumerate(all_moves):
        if not move.is_play:
            legal.append(idx)          # pass (resign is never in all_moves)
            continue
        point = move.point
        if grid.get(point) is not None:
            continue
        # One scan of the neighbours answers the three questions that
        # Board.is_self_capture, Board.will_capture and is_point_an_eye
        # each answer with their own scan.
        has_empty = False          # an empty neighbour = a liberty
        captures = False           # an enemy neighbour string in atari
        friends_in_atari = True    # every friendly neighbour string has 1 liberty
        all_friendly = True        # every neighbour is our stone (eye shape)
        for neighbor in neighbor_table[point]:
            string = grid.get(neighbor)
            if string is None:
                has_empty = True
                all_friendly = False
            elif string.color == player:
                if len(string.liberties) != 1:
                    friends_in_atari = False
            else:
                all_friendly = False
                if len(string.liberties) == 1:
                    captures = True
        # Self-capture: no liberty, no capture, and every friendly string
        # we would join is in atari (Board.is_self_capture).
        if not has_empty and not captures and friends_in_atari:
            continue
        # Ko is only possible when the move captures.
        if captures and state.does_move_violate_ko(player, move):
            continue
        # Own eye (helpers_fast.is_point_an_eye): all neighbours friendly and
        # enough friendly diagonal corners.
        if all_friendly:
            corners = corner_table[point]
            friendly_corners = sum(
                1 for corner in corners
                if (grid.get(corner) is not None
                    and grid.get(corner).color == player))
            off_board = 4 - len(corners)
            if off_board > 0:
                is_eye = off_board + friendly_corners == 4
            else:
                is_eye = friendly_corners >= 3
            if is_eye:
                continue
        legal.append(idx)
    return legal


# tag::node_class_defn[]
class ZeroTreeNode:
# end::node_class_defn[]
    """A search-tree node.

    Branch statistics are stored as NumPy arrays indexed by "slot" (the
    position of a legal move in self.branch_moves), so selecting a branch is
    one vectorized operation instead of a Python loop over every move.
    Children are keyed by slot as well.
    """
# tag::node_class_body[]
    def __init__(self, state, value, priors, parent, last_slot, all_moves,
                 pass_only_when_forced=True):
        """`priors` is the prior vector over all encoder moves (board points +
        pass) and `all_moves[i]` the Move for encoder index i. `last_slot` is
        the slot in `parent` of the move that led here (None for a root).

        With pass_only_when_forced, passing is a branch only when no other
        legal move (that does not fill one of our own eyes) exists. Games are
        then played out to the end, which costs nothing under area scoring
        and keeps self-play from learning to "win" by passing early."""
        self.state = state
        self.value = value
        self.parent = parent                      # <1>
        self.last_slot = last_slot                # <1>
        self.total_visit_count = 1
        legal = legal_non_eye_moves(state, all_moves)
        if pass_only_when_forced:
            plays = [idx for idx in legal if not all_moves[idx].is_pass]
            if plays:
                legal = plays
        self.move_indices = np.array(legal, dtype=np.int64)  # encoder index per slot
        self.branch_moves = [all_moves[idx] for idx in legal]
        self.priors = np.asarray(priors, dtype=np.float64)[self.move_indices]
        self.visits = np.zeros(len(legal), dtype=np.int64)
        self.total_values = np.zeros(len(legal), dtype=np.float64)
        self.children = {}                        # <2>
        self._slots = None

    @property
    def last_move(self):
        if self.parent is None:
            return None
        return self.parent.branch_moves[self.last_slot]

    def moves(self):                              # <3>
        return list(self.branch_moves)            # <3>

    def slot(self, move):
        """Slot of `move`, or None if it is not a legal branch here."""
        if self._slots is None:
            self._slots = {m: k for k, m in enumerate(self.branch_moves)}
        return self._slots.get(move)

    def add_child(self, slot, child_node):        # <4>
        self.children[slot] = child_node          # <4>

    def get_child(self, slot):
        return self.children[slot]

    def has_child(self, slot):                    # <5>
        return slot in self.children              # <5>
# end::node_class_body[]

# tag::node_record_visit[]
    def record_visit(self, slot, value):
        self.total_visit_count += 1
        self.visits[slot] += 1
        self.total_values[slot] += value
# end::node_record_visit[]

    def visit_count_vector(self, num_moves):
        """Visit counts for every encoder move index (0 for illegal moves)."""
        counts = np.zeros(num_moves, dtype=np.int64)
        counts[self.move_indices] = self.visits
        return counts

# tag::node_class_helpers[]
    # Per-move accessors (convenient for inspection; the search uses slots).
    def expected_value(self, move):
        k = self.slot(move)
        if self.visits[k] == 0:
            return 0.0
        return self.total_values[k] / self.visits[k]

    def prior(self, move):
        return self.priors[self.slot(move)]

    def visit_count(self, move):
        k = self.slot(move)
        if k is None:
            return 0
        return int(self.visits[k])
# end::node_class_helpers[]


# tag::zero_defn[]
class ZeroAgent(Agent):
# end::zero_defn[]
    def __init__(self, model, encoder, rounds_per_move=1000, c=2.0,
                 pass_only_when_forced=True, resign_threshold=None,
                 resign_min_moves=30):
        self.model = model
        self.encoder = encoder
        self.pass_only_when_forced = pass_only_when_forced
        # Resign when the searched win probability of the chosen move falls
        # below resign_threshold (None = never; self-play must not resign,
        # since every game has to be played out for training).
        self.resign_threshold = resign_threshold
        self.resign_min_moves = resign_min_moves
        self.last_search = {}

        self.collector = None
        self.temperature = 1.0

        self.num_rounds = rounds_per_move
        self.c = c
        self._infer_fn = None
        # Set to a RemoteInference to evaluate positions on the GPU
        # inference server instead of with self.model.
        self.remote = None
        # Decode every encoder move index once (hot in the search).
        self.all_moves = [encoder.decode_move_index(idx)
                          for idx in range(encoder.num_moves())]
        
    def infer(self, model_input):
        # A compiled tf.function call is several times faster than
        # model.predict / predict_on_batch for single positions in TF2.
        if self._infer_fn is None:
            model = self.model
            self._infer_fn = tf.function(
                lambda x: model(x, training=False), reduce_retracing=True)
        priors, values = self._infer_fn(
            tf.convert_to_tensor(model_input, dtype=tf.float32))
        return priors.numpy(), values.numpy()

    def set_temperature(self, temperature):
        self.temperature = temperature

# tag::zero_select_move_defn[]
    def select_move(self, game_state):
# end::zero_select_move_defn[]
        return self.select_moves([game_state], [self.collector])[0]

    def select_moves(self, game_states, collectors=None, temperatures=None):
        """Run MCTS for several independent positions at once and return one
        move per position.

        Each position gets its own search tree, but in every round the leaf
        positions of all trees are evaluated by the network in a single
        batched call, which is what makes the GPU worthwhile. `collectors`
        is an optional list (one entry per position, may be None) of
        experience collectors to record each root's visit counts into.
        `temperatures` optionally gives each position its own move-sampling
        temperature (default: self.temperature for all).
        """
        return select_moves_multi(
            [(self, game_states, collectors, temperatures)])[0]

    def walk_to_leaves(self, roots):
        """One selection pass down every tree. Terminal positions back up
        their actual result; returns the leaves to expand as (node, slot)."""
        pending = []
        for root in roots:
# tag::zero_walk_down[]
            node = root
            slot = self.select_branch(node)
            while slot is not None and node.has_child(slot):  # <3>
                node = node.get_child(slot)
                slot = self.select_branch(node)
# end::zero_walk_down[]
            if slot is None:
                # Terminal position (game over / no legal moves): back up
                # the actual result instead of expanding further.
                if node.parent is not None:
                    self.back_up(node.parent, node.last_slot,
                                 -1 * self.terminal_value(node.state))
            else:
                pending.append((node, slot))
        return pending

# tag::zero_back_up[]
    @staticmethod
    def back_up(node, slot, value):
        """Record `value` (from the point of view of the player to move at
        `node`) for the branch `slot`, then propagate it up with
        alternating sign."""
        while node is not None:
            node.record_visit(slot, value)
            slot = node.last_slot
            node = node.parent
            value = -1 * value
# end::zero_back_up[]

    def finish_search(self, root, game_state, collector, temperature=None):
        """Record the search result and pick the move to play."""
        visit_counts = root.visit_count_vector(self.encoder.num_moves())
        if visit_counts.sum() == 0:
            # Nothing was searched: the game is already over (or there is no
            # legal move other than filling our own eyes). Passing is the
            # only sensible answer.
            return goboard_fast.Move.pass_turn()
# tag::zero_record_collector[]
        if collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
#             HHY start
            for i in range(8):
                new_root_state_tensor = symmetry_reflection_tensor(board_tensor=root_state_tensor,index=i)
                new_visit_counts = symmetry_reflection_vector(board_vector=visit_counts,index=i,inverse=False)

                collector.record_decision(
                new_root_state_tensor, new_visit_counts)
#             HHY end

# end::zero_record_collector[]

        move = self.sample_move(root, visit_counts, temperature)

        # Search estimate for the chosen move, from our point of view: the
        # mean backed-up value of that branch (or of the root if unvisited).
        slot = root.slot(move)
        if slot is not None and root.visits[slot] > 0:
            value = root.total_values[slot] / root.visits[slot]
        else:
            value = root.total_values.sum() / max(1, root.visits.sum())
        value = float(np.clip(value, -1.0, 1.0))
        win_prob = (value + 1.0) / 2.0
        self.last_search = {
            'player': game_state.next_player.name,
            'value': round(value, 4),
            'win_prob': round(win_prob, 4),
            'network_value': round(float(root.value), 4),
            'visits': int(root.visits.sum()),
        }
        if (self.resign_threshold is not None
                and win_prob < self.resign_threshold
                and move_number(game_state) >= self.resign_min_moves):
            self.last_search['resigned'] = True
            return goboard_fast.Move.resign()
        return move

    def diagnostics(self):
        """Details of the last search (shown by the web demo)."""
        return dict(self.last_search)

    def sample_move(self, root, visit_counts, temperature=None):
        """Pick the move to play from the root's visit counts, sharpened by
        the current temperature."""
# HHY
        move_probs = visit_counts / self.num_rounds
        # Prevent move probs from getting stuck at 0 or 1.
        if temperature is None:
            temperature = self.temperature
        move_probs = np.power(move_probs, 1.0 / temperature)
        move_probs = move_probs / np.sum(move_probs)
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)


        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(self.encoder.num_moves())
        ranked_moves = np.random.choice(
            candidates, self.encoder.num_moves(), replace=False, p=move_probs)
        # Only moves that were search branches are allowed (this excludes
        # illegal moves, filling our own eyes and, unless forced, passing).
        branch_indices = set(root.move_indices.tolist())
        for move_index in ranked_moves:
            if move_index in branch_indices:
                return self.all_moves[move_index]
# HHY


# tag::zero_select_max_visit_count[]
        return root.branch_moves[int(np.argmax(root.visits))]
# end::zero_select_max_visit_count[]

    @staticmethod
    def terminal_value(game_state):
        """+1 if the player to move in a finished game_state has won, else -1."""
        if game_state.is_over():
            result = scoring.compute_game_result(game_state)
            return 1.0 if result.winner == game_state.next_player else -1.0
        # No legal non-eye moves but game not over (should not happen,
        # since passing is always allowed); treat as neutral.
        return 0.0

    def set_collector(self, collector):
        self.collector = collector

# tag::zero_select_branch[]
    def select_branch(self, node):
        """Slot of the branch maximizing Q + c * P * sqrt(N_total) / (n + 1),
        or None if the node has no legal branches. Ties go to the lowest
        slot, as with Python's max() over the moves in order."""
        if len(node.visits) == 0:
            return None
        total_n = node.total_visit_count
        n = node.visits
        q = np.divide(node.total_values, n,
                      out=np.zeros_like(node.total_values), where=n > 0)
        p = node.priors
        score = q + self.c * p * np.sqrt(total_n) / (n + 1)
        return int(np.argmax(score))                           # <1>
# end::zero_select_branch[]

# tag::zero_create_node[]
    def create_node(self, game_state, slot=None, parent=None):
        return self.create_nodes([game_state], [slot], [parent])[0]

    def encode_states(self, game_states):
        """Encode positions for the network: (uint8 planes of shape
        (n, planes, N, N), a random board symmetry index per position)."""
        indices = np.random.randint(8, size=len(game_states))
        planes = np.array([self.encoder.encode(state)
                           for state in game_states], dtype=np.uint8)
        return planes, indices

    def evaluate_encoded(self, planes, indices):
        """Evaluate each position under its symmetry and map the priors back
        to the original orientation. Returns (priors normalised per row,
        values of shape (n,))."""
        if self.remote is not None:
            return self.remote.evaluate(
                [(self.remote.model_key, planes, indices)])[0]
        model_input = np.array([
            symmetry_reflection_tensor(board_tensor=plane, index=index)
            for plane, index in zip(planes, indices)], dtype=np.float32)
        all_priors, all_values = self.infer(model_input)
        priors = np.array([
            symmetry_reflection_vector(board_vector=prior, index=index,
                                       inverse=True)
            for prior, index in zip(all_priors, indices)])
        priors = priors / np.sum(priors, axis=1, keepdims=True)
        return priors, all_values[:, 0]

    def create_nodes(self, game_states, slots=None, parents=None,
                     evaluation=None):
        """Create tree nodes for several positions with one network call.
        slots[k] is the branch of parents[k] that leads to game_states[k].
        `evaluation` is the (priors, values) of these positions if already
        computed (see evaluate_many)."""
        n = len(game_states)
        if slots is None:
            slots = [None] * n
        if parents is None:
            parents = [None] * n
        if evaluation is None:
            evaluation = self.evaluate_encoded(
                *self.encode_states(game_states))
        all_priors, all_values = evaluation

        nodes = []
        for k in range(n):
            priors = all_priors[k]
            # Add Dirichlet noise to the root node.
            if parents[k] is None:
                noise = np.random.dirichlet(
                    0.03 * np.ones_like(priors))
                priors = 0.75 * priors + 0.25 * noise
            value = all_values[k]                              # <2>
            new_node = ZeroTreeNode(
                game_states[k], value, priors,
                parents[k], slots[k], self.all_moves,
                self.pass_only_when_forced)
            if parents[k] is not None:
                parents[k].add_child(slots[k], new_node)
            nodes.append(new_node)
        return nodes
# end::zero_create_node[]

# tag::zero_train[]
    def train(self, experience, learning_rate, batch_size):     # <1>
        num_examples = experience.states.shape[0]

        # Experience is stored in compact integer types; train in float32.
        model_input = experience.states.astype(np.float32)
        visit_counts = experience.visit_counts.astype(np.float32)

        visit_sums = np.sum(                                    # <2>
            visit_counts, axis=1).reshape(                      # <2>
            (num_examples, 1))                                  # <2>
        action_target = visit_counts / visit_sums               # <2>
#         HHY
        # Prevent move probs from getting stuck at 0 or 1.
        # Normalize each example (row) separately, not over the whole batch.
        move_probs = np.power(action_target, 1.0 / self.temperature)
        move_probs = move_probs / np.sum(move_probs, axis=1, keepdims=True)
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs, axis=1, keepdims=True)
        action_target = move_probs
#         HHY
        value_target = experience.rewards.astype(np.float32)

        self.model.compile(
            Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                ),
            loss=['categorical_crossentropy', 'mse'],
            metrics=['accuracy'])
        
#         self.model.summary()
        
        self.model.fit(
            model_input, [action_target, value_target],
            batch_size=batch_size)

# end::zero_train[]

# tag::serialize[] HHY
    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_size'] = self.encoder.board_size
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])
# end::serialize[] HHY
def evaluate_many(agents, state_lists):
    """Network evaluation of state_lists[k] by agents[k] (see
    ZeroAgent.evaluate_encoded). When all agents use the same inference
    server connection, everything goes in a single round trip."""
    encoded = [agent.encode_states(states)
               for agent, states in zip(agents, state_lists)]
    remotes = [agent.remote for agent in agents]
    if remotes and all(r is not None and r.same_connection(remotes[0])
                       for r in remotes):
        return remotes[0].evaluate(
            [(r.model_key, planes, indices)
             for r, (planes, indices) in zip(remotes, encoded)])
    return [agent.evaluate_encoded(planes, indices)
            for agent, (planes, indices) in zip(agents, encoded)]


def select_moves_multi(jobs):
    """Run ZeroAgent.select_moves for several agents in lock-step.

    jobs is a list of (agent, game_states, collectors, temperatures)
    (collectors / temperatures may be None). Every search round, the leaves
    of all agents' trees are evaluated together, so with an inference
    server each round is one request. Returns one list of moves per job.
    """
    jobs = [(agent, list(states),
             list(collectors) if collectors is not None
             else [None] * len(states),
             list(temperatures) if temperatures is not None
             else [agent.temperature] * len(states))
            for agent, states, collectors, temperatures in jobs]
    live = [k for k, job in enumerate(jobs) if job[1]]
    roots = [None] * len(jobs)
    evaluations = evaluate_many([jobs[k][0] for k in live],
                                [jobs[k][1] for k in live])
    for k, evaluation in zip(live, evaluations):
        roots[k] = jobs[k][0].create_nodes(jobs[k][1], evaluation=evaluation)

    num_rounds = max([jobs[k][0].num_rounds for k in live] or [0])
    for i in range(num_rounds):                                # <2>
        expand = []
        for k in live:
            agent = jobs[k][0]
            if i >= agent.num_rounds:
                continue
            pending = agent.walk_to_leaves(roots[k])
            if pending:
                new_states = [node.state.apply_move(node.branch_moves[slot])
                              for node, slot in pending]
                expand.append((k, pending, new_states))
        if not expand:
            continue
        # Expand all leaves with one batched network evaluation.
        evaluations = evaluate_many([jobs[k][0] for k, _, _ in expand],
                                    [states for _, _, states in expand])
        for (k, pending, new_states), evaluation in zip(expand, evaluations):
            agent = jobs[k][0]
            children = agent.create_nodes(
                new_states,
                slots=[slot for _, slot in pending],
                parents=[node for node, _ in pending],
                evaluation=evaluation)
            for (node, slot), child in zip(pending, children):
                agent.back_up(node, slot, -1 * child.value)     # <1>

    results = []
    for k, (agent, states, collectors, temperatures) in enumerate(jobs):
        results.append([
            agent.finish_search(root, state, collector, temperature)
            for root, state, collector, temperature
            in zip(roots[k] or [], states, collectors, temperatures)])
    return results



def move_number(game_state):
    """Number of moves played to reach game_state."""
    n = 0
    while game_state.previous_state is not None:
        n += 1
        game_state = game_state.previous_state
    return n


def load_zero_agent(h5file, load_model=True):
    """Load a ZeroAgent. With load_model=False the network is not built
    (agent.model is None) and agent.remote must be set to a
    RemoteInference that asks the GPU inference server."""
    model = (kerasutil.load_model_from_hdf5_group(h5file['model'])
             if load_model else None)
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_size = h5file['encoder'].attrs['board_size']
    encoder = encoders.get_encoder_by_name(
        encoder_name,
        (board_size,board_size))
    return ZeroAgent(model, encoder, rounds_per_move=100, c=2.0)

# HHY start
# The 8 symmetries of the square board (dihedral group D4), applied to the
# last two axes. Index meanings match the original hand-written loops:
#   0 identity, 1 flip rows, 2 flip cols, 3 rotate 180, 4 transpose,
#   5 rotate 90 clockwise, 6 rotate 90 counter-clockwise, 7 anti-transpose.
def _sym(a, index):
    if index == 0:
        return a
    if index == 1:
        return np.flip(a, axis=-2)
    if index == 2:
        return np.flip(a, axis=-1)
    if index == 3:
        return np.flip(a, axis=(-2, -1))
    if index == 4:
        return np.swapaxes(a, -2, -1)
    if index == 5:
        return np.rot90(a, k=-1, axes=(-2, -1))
    if index == 6:
        return np.rot90(a, k=1, axes=(-2, -1))
    if index == 7:
        return np.flip(np.swapaxes(a, -2, -1), axis=(-2, -1))
    raise ValueError('symmetry index must be in 0..7')


# Only the two 90-degree rotations are not their own inverse.
_INVERSE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 7}


def symmetry_reflection_tensor(board_tensor, index=0):
    """Apply symmetry `index` to a (planes, N, N) board tensor."""
    return np.ascontiguousarray(_sym(np.asarray(board_tensor), index))


def symmetry_reflection_vector(board_vector, index=0, inverse=False):
    """Apply symmetry `index` (or its inverse) to an N*N+1 move vector.
    The final entry (pass) is left unchanged."""
    v = np.asarray(board_vector).flatten()
    n = int(round(np.sqrt(v.size - 1)))
    board = v[:-1].reshape((n, n))
    idx = _INVERSE[index] if inverse else index
    return np.concatenate([_sym(board, idx).flatten(), v[-1:]])

# HHY end


