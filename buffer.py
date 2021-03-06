from collections import deque, namedtuple
import random
from utilities import transpose_list
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "cat_state", "action", "reward", "next_state", "cat_next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, cat_state, action, reward, next_state, cat_next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, cat_state, action, reward, next_state, cat_next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        cat_states = torch.from_numpy(np.vstack([e.cat_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        cat_next_states = torch.from_numpy(np.vstack([e.cat_next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, cat_states, actions, rewards, next_states, cat_next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
