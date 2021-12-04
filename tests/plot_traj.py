import context
import matplotlib.pyplot as plt

from Filter import VisualTraj

traj_data = {
    "stereoGT": "./trajs/offline_mandala0_gt.txt",
    "mono": "./trajs/offline_mandala0_mono.txt",
}

axes = None
for name, filepath in traj_data.items():
    traj = VisualTraj(name, filepath)
    ts = traj.t

    if name == "stereoGT":
        min_t = min(ts)
        max_t = max(ts)
    else:
        min_t = min([min_t, min(ts)])
        max_t = min([max_t, max(ts)])

    axes = traj.plot(axes, min_t, max_t)

plt.legend()
plt.show()
