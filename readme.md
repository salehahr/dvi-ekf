# DVI-EKF
Implementation of a loosely-coupled VI-SLAM.
Based on ![this repo](https://github.com/skrogh/msf_ekf).

## Table of contents
* [Preliminaries/Tests](#preliminariestests)
  * [Offline trajectories](#offline-trajectories)
  * [Fake IMU data](#fake-imu-data)
  * [Reconstructed visual trajectory](#reconstructed-visual-trajectory)
  * [KF propagation only](#kf-propagation-only)
* [Current status (of the filter)](#current-status)

------

## Preliminaries/Tests
### Offline trajectories
```
python3 plot_traj.py
```
![](img/offline_trajs.PNG)

### Fake IMU data
```
python3 plot_imu.py
```

How they were generated:
1. Interpolation of the SLAM pose values between frames.
   [`VisualTraj.interpolate(num_imu_between_frames)`](https://github.com/feudalism/dvi-ekf/blob/291a01af4cdb8d617a4f7a5fb095dc5acd8838bf/Trajectory.py#L161)
2. Numerical differentiation of the interpolated signals.
   [`ImuTraj._gen_unnoisy_imu()`](https://github.com/feudalism/dvi-ekf/blob/291a01af4cdb8d617a4f7a5fb095dc5acd8838bf/Trajectory.py#L249)
    * Straightforward differentiation for x, y, z --> ax, ay, az:
        `np.gradient(interpolated.x, dt)`
    * Converted quaternions to Euler angles:
        `rx, ry, rz = quaternion.as_euler_angles(interpolated.quats)`,
        which are then differentiated to gx, gy, gz

To improve:
- [ ] add bias
- [ ] incorporate effects of gravity
- [ ] get rid of outliers

![](img/offline_noisyimu.PNG)

### Reconstructed visual trajectory
```
python3 plot_reconstructed.py
```

Aim here was to check whether the generation of fake IMU data was correct.
Here I tried to reconstruct the trajectory by integrating the
generated IMU data (without noise).

[`ImuTraj.reconstruct_vis_traj`](https://github.com/feudalism/dvi-ekf/blob/291a01af4cdb8d617a4f7a5fb095dc5acd8838bf/Trajectory.py#L390)
* Gets initial conditions `IC` (for the integration) from original `VisTraj`.
* Integrate:
  * `int_vals = scipy.integrate.cumtrapz(diff_vals, t, initial=0) + IC`.
  * Rotations to quaternions: `quat = quaternion.from_euler_angles(rx, ry, rz)`

![](img/traj_recon.PNG)

#### Debugging ...

![](img/traj_recon_debug.PNG)

### KF propagation only
Using initial pose from monocular trajectory and propagating using IMU values
(not noisy).

![](img/traj_only_prop.PNG)

-----

## Current status (of the filter)
```
python3 main.py
```


Not working, something's wrong...

![](img/kf.PNG)
![](img/kf_z.PNG)
