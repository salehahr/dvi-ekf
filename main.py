from configuration import Config
from Filter import Simulator

from scipy.optimize import basinhopping

## initialise objects
config          = Config()
kf, camera, _   = config.init_filter_objects()

## simulation runs
sim = Simulator(config)
sim.optimise(kf, camera)
# # sim.run(kf, camera)

# ## results
# sim.save_and_plot(camera)