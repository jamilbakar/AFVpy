class OptimizationBasedFalsification:
    def __init__(self, objective, optimizer):
        self.objective = objective
        self.optimizer = optimizer

    def falsify(self, sys, psi):
        def f(x):
            return self.objective(x, sys, psi)
        return self.optimizer(f, sys, psi)