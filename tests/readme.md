# DVI-EKF tests
The Python files here are to be run from the main (parent) folder!

## Table of contents
* [Offline trajectories](#offline-trajectories)
* [Fake IMU data](#fake-imu-data)
* [Reconstructed visual trajectory](#reconstructed-visual-trajectory)
* [Comparing velocities](#comparing-velocities)
* [KF propagation only](#kf-propagation-only)

## Offline trajectories
```
python3 plot_traj.py
```
![](../img/offline_trajs.PNG)

## Fake IMU data
```
python3 plot_imu.py
```

How they were generated:
1. Interpolation of the SLAM pose values between frames.
   [`VisualTraj.interpolate(num_imu_between_frames)`](https://github.com/feudalism/dvi-ekf/blob/master/Filter/Trajectory.py#L162)
2. Numerical differentiation of the interpolated signals.
   [`ImuTraj._gen_unnoisy_imu()`](https://github.com/feudalism/dvi-ekf/blob/master/Filter/Trajectory.py#L247)
    * Straightforward differentiation for x, y, z --> ax, ay, az:
        `np.gradient(interpolated.x, dt)`
    * Converted quaternions to Euler angles
         which are then differentiated to gx, gy, gz

To improve:
- [ ] add bias
- [ ] incorporate effects of gravity
- [ ] get rid of outliers

![](../img/offline_noisyimu.PNG)

## Reconstructed visual trajectory
```
python3 plot_reconstructed.py
```

Aim here was to check whether the generation of fake IMU data was correct.
Here I tried to reconstruct the trajectory by integrating the
generated IMU data (without noise).

[`ImuTraj.reconstruct_vis_traj`](https://github.com/feudalism/dvi-ekf/blob/master/Filter/Trajectory.py#L388)
* Gets initial conditions `IC` (for the integration) from original `VisTraj`.
* Integrate:
  * `int_vals = scipy.integrate.cumtrapz(diff_vals, t, initial=0) + IC`.
  * Rotations to quaternions

![](../img/traj_recon.PNG)


