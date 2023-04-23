# DVI-EKF
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementation of a loosely-coupled VI-ESKF SLAM to estimate
the calibration parameters in a cystoscopic camera-IMU sensor setup.

![](docs/_img/sensor-setup-real.png#gh-light-mode-only)  
![](docs/_img/sensor-setup-real-dark.png#gh-dark-mode-only)  
![](docs/_img/sensor-setup-model.png#gh-light-mode-only)
![](docs/_img/sensor-setup-model-dark.png#gh-dark-mode-only)  

## Usage
1. Install required packages (s. [`requirements.txt`](./requirements.txt) )  
    e.g.
    ```
    pip install -r requirements.txt
    ```  
    e.g. or in conda  
    ```
    conda env create --file requirements.txt
    ```
1. Configure settings in [`config.yaml`](./config.yaml)
2. Run  
    ```
    python3 main.py 
    ```

## Visualisation
![](docs/img/probe_simulation_viz.png)  
