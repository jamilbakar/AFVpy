import numpy as np
def lcb(node, c):
    Qs = [child.Q for child in node.children]
    Ns = [child.N for child in node.children]
    lcbs = [Q - c * np.sqrt(np.log(node.N) / N) for Q, N in zip(Qs, Ns)]
    return node.children[int(np.argmin(lcbs))]