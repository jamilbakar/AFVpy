import numpy as np
from collections import namedtuple
Edge = namedtuple("Edge", ["o", "a", "x"])


def step(sys, s, x):
    o = sys.sensor.observe(s, x.xo)
    a = sys.agent.act(o, x.xa)
    s_next = sys.env.step(s, a, x.xs)
    return o, a, s_next


def lcb(node, c):
    Qs = [child.Q for child in node.children]
    Ns = [child.N for child in node.children]
    lcbs = [Q - c * np.sqrt(np.log(node.N) / N) for Q, N in zip(Qs, Ns)]
    return node.children[int(np.argmin(lcbs))]


class MCTSNode:
    def __init__(self, state, parent, edge, children, N, Q):
        self.state = state
        self.parent = parent
        self.edge = edge            # (o, a, x)
        self.children = children
        self.N = N                  # visit count
        self.Q = Q                  # value estimate


class MCTS:
    def __init__(self, estimate_value, c, k, alpha, select_disturbance, k_max):
        self.estimate_value = estimate_value
        self.c = c                  # exploration constant
        self.k = k                  # progressive widening constant
        self.alpha = alpha          # progressive widening exponent
        self.select_disturbance = select_disturbance
        self.k_max = k_max

    def initialize_tree(self, sys):
        return [MCTSNode(sys.env.initial_distribution().rvs(), None, None, [], 1, 0)]

    def select(self, sys, psi, tree):
        c, k, alpha, node = self.c, self.k, self.alpha, tree[0]
        while len(node.children) > k * node.N ** alpha:
            node = lcb(node, c)
        return node

    def extend(self, sys, psi, tree, node):
        x = self.select_disturbance(sys, node)
        o, a, s_next = step(sys, node.state, x)
        Q = self.estimate_value(sys, psi, s_next)
        snew = MCTSNode(s_next, node, Edge(o, a, x), [], 1, Q)
        node.children.append(snew)
        tree.append(snew)
        while node is not None:
            node.N += 1
            node.Q += (Q - node.Q) / node.N
            Q, node = node.Q, node.parent