
class Specification:
    pass


class STLSpecification(Specification):
    def __init__(self, formula, I):
        self.formula = formula
        self.I = I  # a slice, e.g. slice(2, 10) for Julia 3:10


def evaluate(psi, tau):
    return psi.formula([step.s for step in tau[psi.I]])


def isfailure(psi, tau):
    return not evaluate(psi, tau)