import matplotlib
import matplotlib.pyplot as plt

from Models import create_camera
from tools import Config

matplotlib.use("TkAgg")

config = Config("config.yaml")
camera = create_camera(config)
camera.frames.animate(repeat=True, interval=100, frame="C", dims=camera.traj.lims)
plt.show()
