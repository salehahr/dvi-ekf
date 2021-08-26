from configuration import Config
from Filter import Simulator

## initialise objects
config  = Config()
sim     = Simulator(config)

## tune
# sim.optimise()
sim.optimise_de()

## simulation runs
sim.run(disp_config=True, save_best=True)

## results
sim.save_and_plot()