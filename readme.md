# DVI-EKF
Implementation of a loosely-coupled VI-ESKF SLAM to estimate
the calibration parameters in a camera-IMU sensor probe setup.

[Program outline](https://www.evernote.com/l/AeQSiL2U6txCWbgNAi1G9mUtWune-gjHNlU/)

## Current results
![](img/kf_mono_imu.png)
![](img/kf_mono_cam.png)

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
* [Current results](#current-results)
* [Fake IMU data](#fake-imu-data-generation)
* [Old tests](/old-tests)
