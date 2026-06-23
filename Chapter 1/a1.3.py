class Specification:
    pass

def evaluate(psi, tau):
    raise NotImplementedError  # Julia: abstract evaluate(ψ, τ), defined per spec

def isfailure(psi, tau):
    return not evaluate(psi, tau)