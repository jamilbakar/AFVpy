import numpy as np
from collections import namedtuple
Edge = namedtuple("Edge", ["o", "a", "x"])


def step(sys, s, x):
    o = sys.sensor.observe(s, x.xo)
    a = sys.agent.act(o, x.xa)
    s_next = sys.env.step(s, a, x.xs)
    return o, a, s_next


class RRTNode:
    def __init__(self, state, parent, edge, children, goal_state):
        self.state = state
        self.parent = parent
        self.edge = edge            # (o, a, x)
        self.children = children
        self.goal_state = goal_state


class RRT:
    def __init__(self, sample_goal, compute_objectives, select_disturbance, k_max):
        self.sample_goal = sample_goal
        self.compute_objectives = compute_objectives
        self.select_disturbance = select_disturbance
        self.k_max = k_max

    def initialize_tree(self, sys):
        return [RRTNode(sys.env.initial_distribution().rvs(), None, None, [], None)]

    def select(self, sys, psi, tree):
        sgoal = self.sample_goal(tree)
        objectives = self.compute_objectives(tree, sgoal)
        node = tree[int(np.argmin(objectives))]
        node.goal_state = sgoal
        return node

    def extend(self, sys, psi, tree, node):
        x = self.select_disturbance(sys, node)
        o, a, s_next = step(sys, node.state, x)
        snew = RRTNode(s_next, node, Edge(o, a, x), [], None)
        node.children.append(snew)
        tree.append(snew)