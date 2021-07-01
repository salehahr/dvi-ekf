from Models import RigidSimpleProbe, Camera, Imu

import sys
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def parse_arguments():
    def print_usage():
        print(f"Usage: {__file__} <regen> [<plot>]")
        print("\t <regen>  - regen / noregen")
        print("Optional arguments:")
        print("\t <plot>   - plot")
        sys.exit()

    try:
        do_regenerate = False or (sys.argv[1] == 'regen')
    except:
        print_usage()

    try:
        do_plot = (sys.argv[2] == 'plot')
    except:
        do_plot = False

    return do_regenerate, do_plot
do_regenerate, do_plot = parse_arguments()

# data generation params
num_imu_between_frames = 10

# initialise robot, joint variables
probe_BtoC = RigidSimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)

# parameters from camera
filepath_cam = './trajs/offline_mandala0_gt.txt'
cam = Camera(filepath=filepath_cam, max_vals=50)
cam_interp = cam.interpolate(num_imu_between_frames)
min_t, max_t = cam. t[0], cam.t[-1]

# generate IMU data
filepath_imu = './trajs/offline_mandala0_gt_imugen.txt'
imu = Imu(probe_BtoC, cam_interp)
imu.generate_traj(filepath_imu, do_regenerate)

# reconstruct camera trajectory from IMU data
imu.reconstruct()
imu.traj.reconstructed.name = "imu (B) recon"

# distance
def distance(cam, imu):
    import math
    norm = (cam.x - imu.x)**2 + (cam.z - imu.z)**2 + (cam.z - imu.z)**2
    return [math.sqrt(x) for x in norm]
dist = distance(cam.traj.interpolated, imu.traj.reconstructed)

print(probe_BtoC)

print(f"q: {probe_BtoC.q}")
print(f"q_sym: {probe_BtoC.q_sym}")
print(f"q_dot: {probe_BtoC.q_dot_sym}\n")

if do_plot:
    recon_axes_2d = cam.traj.plot(min_t=min_t, max_t=max_t)
    recon_axes_2d = imu.traj.reconstructed.plot(recon_axes_2d,
            min_t=min_t, max_t=max_t, dist=dist)

    recon_axes_3d = cam.traj.plot_3d()
    recon_axes_3d = imu.traj.reconstructed.plot_3d(ax=recon_axes_3d)

    probe_BtoC.plot(probe_BtoC.q, is_static=True)