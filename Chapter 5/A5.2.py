from collections import namedtuple
Transition = namedtuple("Transition", ["s", "o", "a", "x"])


class TreeSearch:
    pass


def isfailure(psi, tau):
    return not psi.evaluate(tau)


def trajectory(node):
    tau = []
    while node.parent is not None:
        o, a, x = node.edge
        tau.insert(0, Transition(node.parent.state, o, a, x))
        node = node.parent
    return tau


def failures(tree, sys, psi):
    leaves = [n for n in tree if not n.children]
    taus = [trajectory(n) for n in leaves]
    return [t for t in taus if isfailure(psi, t)]


def falsify(alg, sys, psi):
    tree = alg.initialize_tree(sys)
    for i in range(alg.k_max):
        node = alg.select(sys, psi, tree)
        alg.extend(sys, psi, tree, node)
    return failures(tree, sys, psi)