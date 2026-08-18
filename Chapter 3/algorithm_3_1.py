
class Specification:
    pass


class LTLSpecification(Specification):
    def __init__(self, formula):
        self.formula = formula


def evaluate(psi, tau):
    return psi.formula([step.s for step in tau])


def isfailure(psi, tau):
    return not evaluate(psi, tau)