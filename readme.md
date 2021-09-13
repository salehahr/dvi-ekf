# DVI-EKF
Implementation of a loosely-coupled VI-ESKF SLAM to estimate
the calibration parameters in a camera-IMU sensor probe setup.

[Program outline](https://www.evernote.com/l/AeQSiL2U6txCWbgNAi1G9mUtWune-gjHNlU/)

## Notch
### Prop + Update
Tuning in progress
![](img/tuning.png)

### Start at a later frame (260)
`nb`| _  
--- | ---
10 | ![](img/start_260.png)
30 | ![](img/start_260_nb_30.png)
60 | ![](img/start_260_nb_60.png)

### Propagation only  
![](img/notchest_prop_only.png)

* Introduction of constant acceleration model for the notch dofs (imperfect model!)
* Errors of the imperfect model get propagated
* Update stage should try and get the Kalman filter output (blue lines) back to the reference values
    if tuned properly

## Usage
```
python3 main.py -h
```
