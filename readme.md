# DVI-EKF
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)  

Implementation of a loosely-coupled VI-ESKF SLAM to estimate
the calibration parameters in a cystoscopic camera-IMU sensor setup.

![](docs/_img/sensor-setup-real.png)  
![](docs/_img/sensor-setup-model.png)

## Usage
1. Configure settings in `config.yaml`
2. Run  
    ```
    python3 main.py 
    ```
