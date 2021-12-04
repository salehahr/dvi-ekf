from configuration import Config
from Filter import Simulator

config = Config()
sim = Simulator(config)

if config.mode == "tune":
    sim.optimise()
else:
    sim.show_run_progress = False
    sim.run_once()
    # sim.run(disp_config=True, save_best=False)
    sim.save_and_plot(compact=True)
