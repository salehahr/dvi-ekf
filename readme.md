# DVI-EKF
Implementation of a loosely-coupled VI-ESKF SLAM to estimate
the calibration parameters in a camera-IMU sensor probe setup.

[Program outline](https://www.evernote.com/l/AeQSiL2U6txCWbgNAi1G9mUtWune-gjHNlU/)

## Current results
```
python3 main.py <prop/update> nonoise
```

### Simple trajectory
Camera moves in x direction, no rotations.

Propagation only | Propagation + Update
--- | ---
&nbsp;  | `R_p = 1e3`  
&nbsp;  | `R_q = 0.05` 
![](img/kf_transx_prop_imu.png) | ![](img/kf_transx_upd_Rp1000.0_Rq0.05_imu.png)
![](img/kf_transx_prop_cam.png) | ![](img/kf_transx_upd_Rp1000.0_Rq0.05_cam.png)

### Monocular SLAM trajectory
P only  | P + U | P + U | P + U
---     | ---   | ---   | --- |
&nbsp;  | `stdev_a, stdev_om = 1e-3`  | `stdev_a, stdev_om = 1e-3` | `stdev_a, stdev_om = 1e-3`  
&nbsp;  | **`cov_p = 1000`** | **`cov_p = 0.1`**  | **`cov_p = 1e-6`**
&nbsp;  | `cov_q = 0.05` | `cov_q = 0.05` | `cov_q = 0.05`
![](img/kf_mono_prop_imu.png) | ![](img/kf_mono_upd_Rp1000.0_Rq0.05_imu.png) | ![](img/kf_mono_upd_Rp0.1_Rq0.05_imu.png) | ![](img/kf_mono_upd_Rp1e-06_Rq0.05_imu.png)
![](img/kf_mono_prop_cam.png) | ![](img/kf_mono_upd_Rp1000.0_Rq0.05_cam.png) | ![](img/kf_mono_upd_Rp0.1_Rq0.05_cam.png) | ![](img/kf_mono_upd_Rp1e-06_Rq0.05_cam.png)

## Probe
```
python3 vsimpleprobe.py
```
Unconstrained SLAM end | Constrained SLAM end
--- | ---
![](img/probe_uncon.gif) | ![](img/probe_con.gif)

## Table of contents
* [Probe](#probe)
* [Current results](#current-results-propagation-only)
* [Old tests](/old-tests)
