import numpy as np
from collections import deque
def distance_c(node):
    return np.linalg.norm(node.parent.state - node.state)


def distance_h(node, sgoal):
    return np.linalg.norm(sgoal - node.state)


def cost_objectives(tree, sgoal, c=distance_c, h=distance_h):
    costs = {}
    queue = deque([tree[0]])
    while queue:
        node = queue.popleft()
        if node.parent is None:
            costs[node] = 0.0
        else:
            costs[node] = c(node) + costs[node.parent]
        for child in node.children:
            queue.append(child)
    # NOTE: reproduced exactly from the source, which calls h(sgoal, node) even though
    # distance_h is defined as (node, sgoal). Swap the args if using the default h.
    heuristics = [h(sgoal, node) for node in tree]
    objectives = [costs[node] for node in tree]
    return [o + hh for o, hh in zip(objectives, heuristics)]