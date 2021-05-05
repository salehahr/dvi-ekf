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

## Current status (of the filter)
```
python3 main.py all nonoise
```


Not working, something's wrong...

Noise values: `Q = 1e-3`, `Rp = 0.1`, `Rq = 0.05`.

Pictured: KF with both propagation and update steps; **non-noisy IMU**
for the time being.

![](img/kf.PNG)

![](img/kf_zoom1.PNG)

![](img/kf_zoom2.PNG)

## Preliminaries/Tests
### KF propagation only
```
python3 main.py prop nonoise
```
Using initial pose from monocular trajectory and propagating using IMU values
(**not noisy**).

Propagation equations for the states
* ignoring gravity for now, as the fake IMU data does not take
    gravity into account.
* ignoring noise here according to the method suggested in the corresp. paper.
    The noise terms are used in the covariance propagation equations.
![](img/prop_eqns.PNG)

Pictured: after correcting the fake IMU data to be in IMU coordinates. 

![](img/traj_only_prop.PNG)

Increasing `num_imu_between_frames` in [`generate_data.py'](/generate_data.py)
improves the reconstruction accuracy.

![](img/traj_only_prop_incr_imu.PNG)

