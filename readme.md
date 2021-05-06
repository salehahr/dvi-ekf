# DVI-EKF
Implementation of a loosely-coupled VI-SLAM.
Based on ![this repo](https://github.com/skrogh/msf_ekf).

## Table of contents
* [Some notes](#some-notes)
* [Current status (of the filter)](#current-status-of-the-filter)
* [Preliminaries/Tests](#preliminariestests)
  * [Playing with noise values](#playing-with-noise-values)
  * [KF propagation only](#kf-propagation-only)
* [Old tests](/tests)
  * [Offline trajectories](/tests#offline-trajectories)
  * [Fake IMU data](/tests#fake-imu-data)
  * [Reconstructed visual trajectory](/tests#reconstructed-visual-trajectory)
  * [Comparing velocities](/tests#comparing-velocities)

-----

## Some notes
* Propagation stage works as expected,
    s. [this section](#kf-propagation-only)
* [Increasing R values](#playing-with-noise-values)
    --> less trust in camera measurements
    --> convergence towards the green (reference/stereo/IMU) trajectory.
* Possible sources of error in the update stage
    * implementation of quaternions?
    * the generation of the cov matrix P
      * (s. `Filter.propagate_covariance()` [`Filter.py`](/Filter/Filter.py))
      * overview:  
        ![](/img/cov_eqn_overview.png)  
      * detailed definition of the matrices, s.
            [Weiss PhD thesis](http://e-collection.library.ethz.ch/eserv/eth:5889/eth-5889-02.pdf) and
            [Weiss, Siegwart 2011 paper](https://ieeexplore.ieee.org/document/5979982)
      * [my code in Python](https://github.com/feudalism/dvi-ekf/blob/44001bb6960a49e4fe6b42e7dcd5eea7ed4a9952/Filter/Filter.py#L122)
        vs [base repo's code in C++](https://github.com/skrogh/msf_ekf/blob/1bce89fa9125378b932564e0aa0eeaef3bd0ef5a/src/EstimatorBase.cpp#L192)
    * the update equations themselves / the corresp. code?
      * also see the above linked papers for the update equation definitions
      * [my code in Python](https://github.com/feudalism/dvi-ekf/blob/fe038dd593d1f6ac533197f1f6ccb19ee01ca61c/Filter/Filter.py#L155)
      vs [base repo's code in C++](https://github.com/skrogh/msf_ekf/blob/master/src/EstimatorBase.cpp#L273)
    * combination of the above?
* Something funny seems to happen around
    * Frame 210: where the quaternions switch to negative
    * Frame 204: quats seem to start to diverge here. also, up till
        this frame, the y position seems to be following the
        green ref. traj.
* What I've already tried: constraining the quaternions, see
    [constrain_quats](../../tree/constrain_quats) branch.


## Current status (of the filter)
```
python3 main.py all nonoise
```

Not working, something's wrong...

Pictured: KF with both propagation and update steps; **non-noisy IMU**
for the time being.
Noise values: `Q = 1e-3`, `Rp = 0.1`, `Rq = 0.05`.

![](img/kf.PNG)

Zoomed in:

![](img/kf_zoom1.PNG)

## Preliminaries/Tests
### Playing with noise values
#### Increasing Rp
Increasing the measurement noise for the positions:

![](img/rp_sens.png)

#### Increasing Rq
Increasing the measurement noise for the rotations:

![](img/rq_sens.png)

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

