from Filter.Simulator import Simulator
from tools import Config

config = Config("config.yaml")
sim = Simulator(config)

if __name__ == "__main__":
    sim.run_once()
    sim.plot(compact=True)
