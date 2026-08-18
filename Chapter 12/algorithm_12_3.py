# Algorithm 12.3: ODD monitoring via the superlevel set of a distribution. Flags an
# input as in-distribution if its probability density exceeds gamma. dist exposes
# .pdf(input) (e.g. a scipy distribution).


class SuperlevelSetMonitor:
    def __init__(self, dist, gamma):
        self.dist = dist    # distribution
        self.gamma = gamma  # likelihood threshold

    def monitor(self, input):
        return self.dist.pdf(input) > self.gamma