# DVI-EKF
Implementation of a loosely-coupled VI-ESKF SLAM to estimate
the calibration parameters in a camera-IMU sensor probe setup.

[Program outline](https://www.evernote.com/l/AeQSiL2U6txCWbgNAi1G9mUtWune-gjHNlU/)

## Tuning the KF
### Settings
```
py main.py mandala0_mono pu -nc 70 -nb 50 -runs 5
```

```
nc = 70 (half the total frames in mandala0_mono trajectory)
nb = 50 (IMU frequency)
```

![](img/tuning-settings.png)

### Sensitivity analysis `K_p`
1e-4 | 1   | 2
---  | --- | ---
![](img/kf_mandala0_mono_upd_Kp0.0001_Km1.000_compact.png) | ![](img/kf_mandala0_mono_upd_Kp1.0_Km1.000_compact.png)| ![](img/kf_mandala0_mono_upd_Kp2.0_Km1.000_compact.png) |


## Old results
Using `stdev_a, stdev_om = 1e-3`  
Monocular SLAM trajectory  

Prop only plots: [imu](img/kf_mandala0_mono_prop_imu.png) ||
                 [cam](img/kf_mandala0_mono_prop_cam.png)

Only modifying `R_p`:  

**`cov_p = 1000`** | **`cov_p = 0.1`**  | **`cov_p = 1e-3`**
---   | ---   | --- |
`cov_q = 0.5` | `cov_q = 0.5` | `cov_q = 0.5`
![](img/kf_mandala0_mono_upd_Rp1000.0_Rq0.5_imu.png) | ![](img/kf_mandala0_mono_upd_Rp0.1_Rq0.5_imu.png) | ![](img/kf_mandala0_mono_upd_Rp0.001_Rq0.5_imu.png)
![](img/kf_mandala0_mono_upd_Rp1000.0_Rq0.5_cam.png) | ![](img/kf_mandala0_mono_upd_Rp0.1_Rq0.5_cam.png) | ![](img/kf_mandala0_mono_upd_Rp0.001_Rq0.5_cam.png)

Only modifying `R_q`:  

`cov_p = 0.1` | `cov_p = 0.1`  | `cov_p = 0.1`
---   | ---   | --- |
**`cov_q = 1000.0`** | **`cov_q = 0.5`** | **`cov_q = 0.001`**
![](img/kf_mandala0_mono_upd_Rp0.1_Rq1000.0_imu.png) | ![](img/kf_mandala0_mono_upd_Rp0.1_Rq0.5_imu.png) | ![](img/kf_mandala0_mono_upd_Rp0.1_Rq0.001_imu.png)
![](img/kf_mandala0_mono_upd_Rp0.1_Rq1000.0_cam.png) | ![](img/kf_mandala0_mono_upd_Rp0.1_Rq0.5_cam.png) | ![](img/kf_mandala0_mono_upd_Rp0.1_Rq0.001_cam.png)

## Usage
```
python3 main.py -h
```
