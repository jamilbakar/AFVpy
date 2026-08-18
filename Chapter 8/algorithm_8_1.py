# Algorithm 8.1: a specification that checks whether a trajectory avoids a given set.
# The set can be any object supporting membership (`in`); Julia uses LazySets.jl.

class Specification:
    pass


class AvoidSetSpecification(Specification):
    def __init__(self, set):
        self.set = set  # avoid set


def evaluate(psi, tau):
    return all(step.s not in psi.set for step in tau)  # Julia: step.s ∉ ψ.set