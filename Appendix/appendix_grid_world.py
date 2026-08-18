import numpy as np


class Categorical:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)

    def rvs(self):
        return np.random.choice(len(self.probs), p=self.probs)  # 0-based index


class SetCategorical:
    def __init__(self, values, probs=None):
        self.values = values
        self.probs = None if probs is None else np.asarray(probs, dtype=float)

    def rvs(self):
        idx = np.random.choice(len(self.values), p=self.probs)
        return self.values[idx]


class GridWorld:
    def __init__(self, size=(10, 10), terminal_states=None,
                 directions=None, tprob=0.7):
        self.size = size                                  # dimensions of the grid
        self.terminal_states = terminal_states or [[5, 5], [7, 8]]  # goal and obstacle
        # up, down, left, right
        self.directions = directions or [[0, 1], [0, -1], [-1, 0], [1, 0]]
        self.tprob = tprob                                # probability do not slip

    def Ds(self, s, a):
        slip_prob = (1 - self.tprob) / (len(self.directions) - 1)
        probs = np.full(len(self.directions), slip_prob)
        probs[a] = self.tprob  # a is 0-based here (Julia is 1-based)
        return Categorical(probs)

    def step(self, s, a, x=None):
        if x is None:
            x = self.Ds(s, a).rvs()
        if list(s) in self.terminal_states:
            return s
        dir = self.directions[x]
        return np.clip(np.array(s) + np.array(dir), [1, 1], self.size)

    def initial_distribution(self):
        return SetCategorical([[1, 1]])


class DiscreteAgent:
    def __init__(self, policy):
        self.policy = policy  # dictionary mapping states to actions

    def act(self, o):
        return self.policy[o]