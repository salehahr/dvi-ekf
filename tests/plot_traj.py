import context
import copy
import numpy as np
np.random.seed(0)

import scipy as sp

import matplotlib.pyplot as plt
from Filter import VisualTraj
from Filter import Quaternion

traj = VisualTraj('cam gt', './trajs/mandala0_mono.txt')
ts = traj.t

pos_labels = ['x', 'y', 'z']
quat_labels = ['qx', 'qy', 'qz', 'qw']

traj_noisy = VisualTraj('cam noisy', './trajs/mandala0_mono_noisy.txt')
_, axes = plt.subplots(4, 2)

def generate_noisy_data():
    ## generate noisy cam positions
    traj_noisy.t = np.copy(traj.t)

    pos_n = np.random.normal(loc=0., scale=0.005, size=(3, len(ts)))
    pos_drift = sp.integrate.cumtrapz(pos_n, ts, initial=0)
    for i, label in enumerate(pos_labels):
        traj_noisy.__dict__[label] = np.copy(traj.__dict__[label]) + pos_drift[i,:]

    ## generate noisy cam rotations
    rot_n = np.random.normal(loc=0., scale=0.003, size=(3, len(ts)))
    rot_drift = sp.integrate.cumtrapz(rot_n, ts, initial=0,)

    # init. quat attributes
    traj_noisy.quats = [None] * len(ts)
    for label in quat_labels:
        traj_noisy.__dict__[label] = [None] * len(ts)

    for i in range(len(ts)):
        eul = rot_drift[:,i]
        error_quat = Quaternion(val=eul, euler='xyz')
        traj_noisy.quats[i] = traj.quats[i] * error_quat

        # generate quat element arrays
        for label in quat_labels:
            traj_noisy.__dict__[label][i] = traj_noisy.quats[i].__dict__[label[-1]]

    ## save to text file
    traj_noisy.write_to_file(traj_noisy.filepath)
    sys.exit()

# generate_noisy_data()

## plotting
for i, label in enumerate(pos_labels):
    axes[i+1][0].plot(ts, traj.__dict__[label], label=traj.name)
    axes[i+1][0].plot(ts, traj_noisy.__dict__[label], label=traj_noisy.name)
    axes[i+1][0].set_title(label)
    axes[i+1][0].legend()

for i, label in enumerate(quat_labels):
    axes[i][1].plot(ts, traj.__dict__[label], label=traj.name)
    axes[i][1].plot(ts, traj_noisy.__dict__[label], label=traj_noisy.name)
    axes[i][1].set_title(label)
    axes[i][1].legend()




# axes[1][0].plot(ts, traj.x)
# axes[2][0].plot(ts, traj.y)
# axes[3][0].plot(ts, traj.z)

# axes[0][1].plot(ts, traj.qw)
# axes[1][1].plot(ts, traj.qx)
# axes[2][1].plot(ts, traj.qy)
# axes[3][1].plot(ts, traj.qz)
    
plt.legend()
plt.show()