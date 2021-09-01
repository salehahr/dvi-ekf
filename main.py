from configuration import Config
from Filter import Simulator

## initialise objects
config  = Config()
sim     = Simulator(config)

if config.mode == 'tune':
    sim.optimise()
else:
    sim.show_run_progress = False
    sim.run(disp_config=True, save_best=True)
    sim.save_and_plot(compact=True)