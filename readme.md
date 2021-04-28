# DVI-EKF
Implementation of a loosely-coupled VI-SLAM.
Based on ![this repo](https://github.com/skrogh/msf_ekf).

## Table of contents
* [Preliminaries/Tests](#preliminariestests)
  * [Offline trajectories](#offline-trajectories)
  * [Fake IMU data](#fake-imu-data)
  * [Reconstructed visual trajectory](#reconstructed-visual-trajectory)
  * [Comparing velocities (**NEW**)](#comparing-velocities)
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

![](img/offline_noisyimu.PNG)

### Reconstructed visual trajectory
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

![](img/traj_recon.PNG)

### Comparing velocities
Tried to compare
* velocities from the stereo trajectory (from numerical differentation of x, y, z)
* velocities from kalman filter (propagation only, IMU without noise)
![](img/velocities.png)

Corrected the offset by setting the initial values v0 in the Kalman filter
to the initial values from the stereo trajectory.

(Previously `v0 = [0., 0., 0.]`, now `v0 = [stereoGT_traj.vx[0], stereoGT_traj.vy[0], stereoGT_traj.vz[0]]`)

![](img/velocities_corrected.png)

### KF propagation only
Using initial pose from monocular trajectory and propagating using IMU values
(**not noisy**).

Currently, propagation equations seem to check out for the quaternions!
Pictured: after correcting the initial values for the velocity `v0`:

![](img/traj_only_prop.PNG)

-----

## Current status (of the filter)
```
python3 main.py
```


Not working, something's wrong...

Pictured: KF with both propagation and update steps; **non-noisy IMU**
for the time being.

![](img/kf.PNG)
![](img/kf_z.PNG)
