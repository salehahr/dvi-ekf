import context

import matplotlib.pyplot as plt
from Filter import VisualTraj
from Models import Camera

traj_data = {
            # 'stereoGT': "./trajs/offline_mandala0_gt.txt",
            'miro' : "./trajs/mirocam.txt",
            }

axes = None
for name, filepath in traj_data.items():
    traj = VisualTraj(name, filepath)
    ts = traj.t
    
    if name == 'stereoGT' or len(traj_data) == 1:
        min_t = min(ts)
        max_t = max(ts)
    else:
        min_t = min([min_t, min(ts)])
        max_t = min([max_t, max(ts)])
    
    _, axes = plt.subplots(4, 2)
    axes[1][0].plot(ts, traj.x)
    axes[2][0].plot(ts, traj.y)
    axes[3][0].plot(ts, traj.z)

    axes[0][1].plot(ts, traj.qw)
    axes[1][1].plot(ts, traj.qx)
    axes[2][1].plot(ts, traj.qy)
    axes[3][1].plot(ts, traj.qz)
    # axes = traj.plot(axes, min_t, max_t)
    
plt.legend()
plt.show()