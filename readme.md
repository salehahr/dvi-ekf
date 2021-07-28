# DVI-EKF
Implementation of a loosely-coupled VI-ESKF SLAM to estimate
the calibration parameters in a camera-IMU sensor probe setup.

[Program outline](https://www.evernote.com/l/AeQSiL2U6txCWbgNAi1G9mUtWune-gjHNlU/)

## Simple camera trajectory
```
python3 cam_fake_traj.py # for generating a simple trajectory
python3 simple_robot.py <regen/noregen> <plot/noplot>
```

Camera moves in x  
![](img/fakecam-x.png)

Camera moves in y  
![](img/fakecam-y.png) 

Camera moves in z  
![](img/fakecam-z.png)

Camera rotates around its x-axis (90 deg)  
![](img/fakecam-rx-90.png)

Camera rotates around its x-axis (270 deg)  
![](img/fakecam-rx-270-qwnotpos.png)

Camera rotates around its x-axis (270 deg -- constrained `q_w` to be positive)  
![](img/fakecam-rx-270-qwpos.png)

## Filter
![](img/kf.png)

- [ ] Shown are IMU positions and rotations, I still need to work on
proper plot functions for the rest of the states.
- [ ] Equations/Jacobian stuff still needs checking.

## Probe
```
python3 vsimpleprobe.py
```
Unconstrained SLAM end | Constrained SLAM end
--- | ---
![](img/probe_uncon.gif) | ![](img/probe_con.gif)

## Fake IMU data generation
```
python3 simple_robot.py regen plot
```
![](img/gen_fake_imu_validation.PNG)

Generated fake IMU data based on kinematics relations between
the camera and the IMU. Shown here: for the first 50 camera values.

![](img/gen_fake_imu.PNG)

## Table of contents
* [Probe](#probe)
* [Simple camera trajectory](#simple-camera-trajectory)
* [Filter](#filter)
* [Fake IMU data](#fake-imu-data-generation)
* [Old tests](/old-tests)
