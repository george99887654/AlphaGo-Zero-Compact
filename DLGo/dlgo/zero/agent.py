import numpy as np
from keras.optimizers import SGD
from keras.optimizers import Adam

from .. import kerasutil
from .. import goboard_fast
from .. import encoders

from dlgo.agent.helpers_fast import is_point_an_eye 

from ..agent import Agent

__all__ = [
    'ZeroAgent',
    'load_zero_agent',
]

# tag::branch_struct[]
class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
# end::branch_struct[]


# tag::node_class_defn[]
class ZeroTreeNode:
# end::node_class_defn[]
# tag::node_class_body[]
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent                      # <1>
        self.last_move = last_move                # <1>
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if state.is_valid_move(move) and not is_point_an_eye(state.board, move.point, state.next_player):
                self.branches[move] = Branch(p)
        self.children = {}                        # <2>

    def moves(self):                              # <3>
        return self.branches.keys()               # <3>

    def add_child(self, move, child_node):        # <4>
        self.children[move] = child_node          # <4>

    def has_child(self, move):                    # <5>
        return move in self.children              # <5>
# end::node_class_body[]

# tag::node_record_visit[]
    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value
# end::node_record_visit[]

# tag::node_class_helpers[]
    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
# end::node_class_helpers[]


# tag::zero_defn[]
class ZeroAgent(Agent):
# end::zero_defn[]
    def __init__(self, model, encoder, rounds_per_move=1000, c=2.0):
        self.model = model
        self.encoder = encoder

        self.collector = None
        self.temperature = 1.0

        self.num_rounds = rounds_per_move
        self.c = c
        
    def set_temperature(self, temperature):
        self.temperature = temperature

# tag::zero_select_move_defn[]
    def select_move(self, game_state):
# end::zero_select_move_defn[]
# tag::zero_walk_down[]
        root = self.create_node(game_state)           # <1>

        for i in range(self.num_rounds):              # <2>
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):          # <3>
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
# end::zero_walk_down[]

# tag::zero_back_up[]
            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(
                new_state, parent=node)

            move = next_move
            value = -1 * child_node.value             # <1>
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value
# end::zero_back_up[]

# tag::zero_record_collector[]
        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(
                    self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
#             HHY start
            for i in range(8):
                new_root_state_tensor = symmetry_reflection_tensor(board_tensor=root_state_tensor,index=i)
                new_visit_counts = symmetry_reflection_vector(board_vector=visit_counts,index=i,inverse=False)

                self.collector.record_decision(
                new_root_state_tensor, new_visit_counts)
#             HHY end          
            
# end::zero_record_collector[]

# HHY
        move_probs = np.array([
                        root.visit_count(
                            self.encoder.decode_move_index(idx))/self.num_rounds
                        for idx in range(self.encoder.num_moves())
                     ])
        # Prevent move probs from getting stuck at 0 or 1.
        move_probs = np.power(move_probs, 1.0 / self.temperature)
        move_probs = move_probs / np.sum(move_probs)
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)
        

        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(self.encoder.num_moves())
        ranked_moves = np.random.choice(
            candidates, self.encoder.num_moves(), replace=False, p=move_probs)
        for move_index in ranked_moves:
            move = self.encoder.decode_move_index(move_index)
            if game_state.is_valid_move(move):
                return move
# HHY


# tag::zero_select_max_visit_count[]
        return max(root.moves(), key=root.visit_count)
# end::zero_select_max_visit_count[]

    def set_collector(self, collector):
        self.collector = collector

# tag::zero_select_branch[]
    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
  
            return q + self.c * p * np.sqrt(total_n) / (n + 1)

        return max(node.moves(), key=score_branch)             # <1>
# end::zero_select_branch[]

# tag::zero_create_node[]
    def create_node(self, game_state, move=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
#         HHY        
        index = np.random.randint(8)
        new_state_tensor = symmetry_reflection_tensor(board_tensor=state_tensor,index=index)
        model_input = np.array([new_state_tensor])          # <1>
        priors, values = self.model.predict(model_input)
        priors = priors[0]                                     # <2>
       
        priors_tmp = symmetry_reflection_vector(board_vector=priors,index=index,inverse=True)    
        priors = priors_tmp
        
        priors = priors / np.sum(priors)
        

#         HHY

        # Add Dirichlet noise to the root node.
        if parent is None:
            noise = np.random.dirichlet(
                0.03 * np.ones_like(priors))
            priors = 0.75 * priors + 0.25 * noise
        value = values[0][0]                                   # <2>
        move_priors = {                                        # <3>
            self.encoder.decode_move_index(idx): p             # <3>
            for idx, p in enumerate(priors)                    # <3>
        }                                                      # <3>
        new_node = ZeroTreeNode(
            game_state, value,
            move_priors,
            parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node
# end::zero_create_node[]

# tag::zero_train[]
    def train(self, experience, learning_rate, batch_size):     # <1>
        num_examples = experience.states.shape[0]

        model_input = experience.states

        visit_sums = np.sum(                                    # <2>
            experience.visit_counts, axis=1).reshape(           # <2>
            (num_examples, 1))                                  # <2>
        action_target = experience.visit_counts / visit_sums    # <2>
#         HHY
        # Prevent move probs from getting stuck at 0 or 1.
        move_probs = np.power(action_target, 1.0 / self.temperature)
        move_probs = move_probs / np.sum(move_probs)
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)
        action_target = move_probs
#         HHY
        value_target = experience.rewards

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

def load_zero_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_size = h5file['encoder'].attrs['board_size']
    encoder = encoders.get_encoder_by_name(
        encoder_name,
        (board_size,board_size))
    return ZeroAgent(model, encoder, rounds_per_move=100, c=2.0)

# HHY start

def symmetry_reflection_tensor(board_tensor,index=0):    
    new_state_tensor = np.zeros(np.shape(board_tensor))
    

    if index == 0:
        new_state_tensor = board_tensor  
        
    elif index == 1:
        for r in range(9):
            for c in range(9):
                new_state_tensor[:,r,c] = board_tensor[:,9-r-1,c]

    elif index == 2:
        for r in range(9):
            for c in range(9):
                new_state_tensor[:,r,c] = board_tensor[:,r,9-c-1]        

    elif index == 3:        
        for r in range(9):
            for c in range(9):
                new_state_tensor[:,r,c] = board_tensor[:,9-r-1,9-c-1]        
        
    elif index == 4:        
        for r in range(9):
            for c in range(9):
                new_state_tensor[:,r,c] = board_tensor[:,c,r]        
        
    elif index == 5:        
        for r in range(9):
            for c in range(9):
                new_state_tensor[:,r,c] = board_tensor[:,9-c-1,r]        
        
    elif index == 6:        
        for r in range(9):
            for c in range(9):
                new_state_tensor[:,r,c] = board_tensor[:,c,9-r-1]
        
    elif index == 7:        
        for r in range(9):
            for c in range(9):
                new_state_tensor[:,r,c] = board_tensor[:,9-c-1,9-r-1]
    
            
    return new_state_tensor
    
    
    
def symmetry_reflection_vector(board_vector,index=0,inverse=False):
    """
    5,6 swap
    """
#     print(board_vector)
    board_vector2 = board_vector.flatten()
    v_tmp = board_vector2[:-1]
    v_pass = board_vector2[-1:]
    tmp = v_tmp.reshape((1,9,9))
    new_tmp = np.zeros((1,9,9))

    if index == 0:
        new_tmp = tmp  
        
    elif index == 1:
        for r in range(9):
            for c in range(9):
                new_tmp[:,r,c] = tmp[:,9-r-1,c]

    elif index == 2:
        for r in range(9):
            for c in range(9):
                new_tmp[:,r,c] = tmp[:,r,9-c-1]        

    elif index == 3:        
        for r in range(9):
            for c in range(9):
                new_tmp[:,r,c] = tmp[:,9-r-1,9-c-1]        
        
    elif index == 4:        
        for r in range(9):
            for c in range(9):
                new_tmp[:,r,c] = tmp[:,c,r]        
        
    elif index == 5 and inverse == True:        
        for r in range(9):
            for c in range(9):
                new_tmp[:,r,c] = tmp[:,c,9-r-1]
                
    elif index == 5 and inverse == False:        
        for r in range(9):
            for c in range(9):
                new_tmp[:,r,c] = tmp[:,9-c-1,r]
                
    elif index == 6 and inverse == True:        
        for r in range(9):
            for c in range(9):
                new_tmp[:,r,c] = tmp[:,9-c-1,r]
                
    elif index == 6 and inverse == False:        
        for r in range(9):
            for c in range(9):
                new_tmp[:,r,c] = tmp[:,c,9-r-1]
        
    elif index == 7:        
        for r in range(9):
            for c in range(9):
                new_tmp[:,r,c] = tmp[:,9-c-1,9-r-1]
                
    v_tmp = new_tmp.flatten()        
    v = np.concatenate([v_tmp,v_pass])
    vector = v.reshape(9*9+1)
#     print(v)

    return vector
    
# HHY end


