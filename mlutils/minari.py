import logging

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
try:
    import gymnasium as gym
    import minari

    def load_env_maker(config, render=False):
        env_name = config.env_name
        if env_name == "halfcheetah":
            env_name = "HalfCheetah-v5"
        render_mode = "rgb_array" if render else None

        def env_maker():
            return gym.make(env_name, render_mode=render_mode)

        return env_maker

    class IterableEpisodeData(torch.utils.data.Dataset):
        def __init__(self, episode_data: minari.EpisodeData):
            self.data = episode_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data.observations[index], self.data.actions[index]

    class SequentialDataset(torch.utils.data.Dataset):
        """Dataset for sequential data. Takes a  minari dataset and returns a dataset
        that returns a tuple of torch tensors with size ([state_history,
        observation_size, action_length, action_size) for each index. These values
        correspond to the N=state_history previous states, the current state and
        the next N=action_length actions. If you set seq_first=False, the order
        of the returned tuple will be ([observation_size, state_history],
        [action_size, action_length])
        """

        def __init__(self, dataset, state_history, action_length, seq_first=False):
            self.state_history = state_history
            self.action_length = action_length
            self.observations = []  # List of observations
            self.actions = []  # List of actions
            self.nonvalid_idx = []
            self.valid_idx = []
            idx = 0
            for episode in tqdm(dataset.iterate_episodes()):
                elen = len(episode)
                self.observations.append(
                    episode.observations[:elen]
                )  # observations include *next_state* in env.step, so is elen+1!
                self.actions.append(episode.actions)
                nonvalid_idx = [
                    *range(idx, idx + state_history - 1),  # The beginning
                    *range(idx + elen - action_length + 1, idx + elen),  # the end
                ]
                self.valid_idx.extend(
                    [i for i in range(idx, idx + elen) if i not in nonvalid_idx]
                )
                self.nonvalid_idx.extend(nonvalid_idx)
                idx += elen
            self.observations = np.concatenate(self.observations, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            self.observations = torch.from_numpy(self.observations).float()
            self.actions = torch.from_numpy(self.actions).float()
            self.seq_first = seq_first

    def __len__(self):
        return len(self.valid_idx)
        # return self.observations.shape[0]

    def __getitem__(self, idx):
        idx = self.valid_idx[idx]
        # if idx in self.nonvalid_idx:
        #    idx = np.random.choice(self.valid_idx)
        observation = self.observations[(idx - self.state_history + 1) : idx + 1]
        action = self.actions[idx : idx + self.action_length]
        if not self.seq_first:
            observation = observation.permute(1, 0)
            action = action.permute(1, 0)
        return observation, action

except ImportError:
    logger.error(
        "minari is not installed. Please install it with `pip install minari`."
    )
