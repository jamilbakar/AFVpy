from functools import singledispatch

class Specification:
    pass

@singledispatch
def evaluate(psi, tau):
    raise NotImplementedError 

def isfailure(psi, tau):
    return not evaluate(psi, tau)