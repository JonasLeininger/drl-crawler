import random
import numpy as np
from collections import namedtuple, deque

import torch

class ReplayBuffer(object):

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def sample(self):
        """
        Sample a minibatch from memory uniformly
        :return:
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)