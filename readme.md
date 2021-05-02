# DVI-EKF
Implementation of a loosely-coupled VI-SLAM.
Based on ![this repo](https://github.com/skrogh/msf_ekf).

## Table of contents
* [Current status (of the filter)](#current-status-of-the-filter)
* [Preliminaries/Tests](#preliminariestests)
  * [KF propagation only](#kf-propagation-only)
* [Old tests](/tests)
  * [Offline trajectories](/tests#offline-trajectories)
  * [Fake IMU data](/tests#fake-imu-data)
  * [Reconstructed visual trajectory](/tests#reconstructed-visual-trajectory)
  * [Comparing velocities](/tests#comparing-velocities)

-----

## Preliminaries/Tests
### KF propagation only
```
python3 main.py prop nonoise
```
Using initial pose from monocular trajectory and propagating using IMU values
(**not noisy**).

Currently, propagation equations seem to check out for the quaternions!
Pictured: after correcting the initial values for the velocity `v0`:

![](img/traj_only_prop.PNG)

## Current status (of the filter)
```
python3 main.py all nonoise
```


Not working, something's wrong...

Pictured: KF with both propagation and update steps; **non-noisy IMU**
for the time being.

![](img/kf.PNG)
![](img/kf_zoom.PNG)
