import numpy as np

# Compact storage types. Board planes from the zero encoder are 0/1, visit
# counts are bounded by the MCTS rounds per move, rewards are +1/-1. Storing
# these instead of float64 makes experience files ~8x smaller; training
# converts to float32 when it uses them.
STATE_DTYPE = np.uint8
VISIT_DTYPE = np.uint16
REWARD_DTYPE = np.int8

__all__ = [
    'ZeroExperienceCollector',
    'ZeroExperienceBuffer',
    'combine_experience',
    'load_experience',
]


# tag::zero_experience_collector[]
class ZeroExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(
            np.asarray(state).astype(STATE_DTYPE))
        self._current_episode_visit_counts.append(
            np.asarray(visit_counts).astype(VISIT_DTYPE))

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_visit_counts = []
# end::zero_experience_collector[]


class ZeroExperienceBuffer:
    def __init__(self, states, visit_counts, rewards):
        self.states = np.asarray(states).astype(STATE_DTYPE, copy=False)
        self.visit_counts = np.asarray(visit_counts).astype(VISIT_DTYPE, copy=False)
        self.rewards = np.asarray(rewards).astype(REWARD_DTYPE, copy=False)

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset(
            'states', data=self.states)
        h5file['experience'].create_dataset(
            'visit_counts', data=self.visit_counts)
        h5file['experience'].create_dataset(
            'rewards', data=self.rewards)


def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])

    return ZeroExperienceBuffer(
        combined_states,
        combined_visit_counts,
        combined_rewards)


def load_experience(h5file):
    return ZeroExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        visit_counts=np.array(h5file['experience']['visit_counts']),
        rewards=np.array(h5file['experience']['rewards']))