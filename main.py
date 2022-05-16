from Filter import Simulator
from tools import Config, SimMode

config = Config("config.yaml")
sim = Simulator(config)

if config.sim.mode == SimMode.TUNE:
    sim.optimise()
else:
    sim.run_once()
    sim.plot(compact=True)
