class TrajectoryDistribution:
    pass


def initial_state_distribution(p):
    raise NotImplementedError  # distribution over initial states


def disturbance_distribution(p, t):
    raise NotImplementedError  # disturbance distribution at time t


def depth(p):
    raise NotImplementedError  # number of time steps in sampled trajectories