import roboticstoolbox as rtb
from spatialmath import SE3

import sympy as sp

from roboticstoolbox.backends.PyPlot import PyPlot
import tkinter

# just so that the plot is orientated correctly...
plot_rotation = SE3.Rx(0, 'deg') * SE3.Ry(0, 'deg')

class SimpleProbe(rtb.DHRobot):
    """ Base coordinates are the IMU coordinates. """

    def __init__(self, scope_length, theta_cam):
        imu_cam = ImuToCam()
        cam_slam = CamToSlam(scope_length=scope_length, theta_cam=theta_cam)

        probe = imu_cam + cam_slam

        super().__init__(probe.links, name='probe', base=plot_rotation)

    def plot(self, config, dt=0.05, block=True, limits=None, movie=None):
        env = PyPlot()

        # visuals
        limit_x = [-0.1, 0.]
        limit_y = [-0.3, 0.2]
        limit_z = [-0.15, 0.05]
        limits = [*limit_x, *limit_y, *limit_z] if (limits is None) else limits

        env.launch(limits=limits)
        ax = env.fig.axes[0]
        # azim, elev = -78, 129 # 3d view
        # azim, elev = -114, 144 # 3d view
        # azim, elev = 0, 0 # zy plane
        # azim, elev = 0, 90 # zx plane
        try:
            ax.view_init(azim=azim, elev=elev)
        except NameError:
            pass # default view

        # robots
        env.add(self, jointlabels=True, jointaxes=False,
                    eeframe=True, shadow=False)

        # save gif
        loop = True if (movie is None) else False
        images = []

        try:
            while True:
                for qk in config:
                    self.q = qk
                    env.step(dt)

                    if movie is not None:
                        images.append(env.getframe())

                if movie is not None:
                    # save it as an animated gif
                    images[0].save(
                        movie,
                        save_all=True,
                        append_images=images[1:],
                        optimize=False,
                        duration=dt,
                        loop=0,
                    )
                if not loop:
                    break

            if block:
                env.hold()

        except tkinter.TclError:
            # handles error when closing the window
            return None

class ImuToCam(rtb.DHRobot):
    """ Base coordinates are the IMU coordinates. """
    def __init__(self):
        links = [
            rtb.RevoluteDH(d=0.05, a=0, alpha=sp.pi/2, offset=0),
            # rtb.PrismaticDH(theta=0, a=0.05, alpha=0, offset=0),
            ]
        super().__init__(links, name='imu_cam')

class CamToSlam(rtb.DHRobot):
    """ Base coordinates are the real camera coordinates. """
    def __init__(self, scope_length, theta_cam):
        links = self._gen_links(scope_length, theta_cam)
        super().__init__(links, name='cam_slam')

    def _gen_links(self, scope_length, theta_cam):
        return [
            # cam to scope end
            rtb.RevoluteDH(d=scope_length, a=0, alpha=theta_cam, offset=0),
            # socpe end to slam
            rtb.RevoluteDH(d=0*scope_length, a=0, alpha=0, offset=0),
            ]

