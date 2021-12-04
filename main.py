from configuration import Config
from Filter import Simulator

config = Config()
sim = Simulator(config)

if config.sim.mode == "tune":
    sim.optimise()
else:
    sim.show_run_progress = False
    sim.run_once()
    sim.save_and_plot(compact=True)
