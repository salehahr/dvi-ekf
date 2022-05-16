from Filter import Simulator
from tools import Config, SimMode

config = Config("config.yaml")
sim = Simulator(config)

if config.sim.mode == SimMode.TUNE:
    sim.optimise()
else:
    sim.show_run_progress = False
    sim.run_once()
    sim.plot(compact=True)
