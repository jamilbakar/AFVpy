class Agent:
    pass

class Environment:
    pass

class Sensor:
    pass

class System:
    def __init__(self, agent, env, sensor):
        self.agent = agent
        self.env = env
        self.sensor = sensor