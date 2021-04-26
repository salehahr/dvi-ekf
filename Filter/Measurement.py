import numpy as np
import quaternion

class Measurement(object):
    """ Base measurement class.

    Attributes:
        t       time
        vec     6x1 measurement vector
    """

    def __init__(self, t, v1, v2):
        self.t = t
        self.vec = np.hstack((v1, v2))

class VisualMeasurement(Measurement):
    """ Visual measurements from SLAM
    containing position and orientation. """

    def __init__(self, t, pos, rot):
        super().__init__(t, pos, rot)

        self.pos = pos
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

        self.rot = rot
        self.qx = rot[0]
        self.qy = rot[1]
        self.qz = rot[2]
        self.qw = rot[3]
        self.qrot = np.quaternion(self.qw, self.qx, self.qy, self.qz)

class ImuMeasurement(Measurement):
    """ IMU measurements containing acceleration and angular velocity.
    Can either be a measurement at a single time instance
    or a queue of measurements
    """

    def __init__(self, t, acc, om):
        super().__init__(t, acc, om)

        self.acc = acc
        self.ax = acc[0]
        self.ay = acc[1]
        self.az = acc[2]

        self.om = om
        self.gx = om[0]
        self.gy = om[1]
        self.gz = om[2]

        if self.t.ndim == 0:
            self._is_queue = False
        elif self.t.ndim > 0:
            self._is_queue = True

    def at_index(self, index):
        """ Returns single IMU measurement at the given index. """

        if not self._is_queue:
            return self

        t = self.t[index]

        ax = self.ax[index]
        ay = self.ay[index]
        az = self.az[index]        
        acc = np.array([ax, ay, az])

        gx = self.gx[index]
        gy = self.gy[index]
        gz = self.gz[index]
        om = np.array([gx, gy, gz])

        return ImuMeasurement(t, acc, om)

    @property
    def is_queue(self):
        return self._is_queue

    @is_queue.setter
    def is_queue(self, bval):
        self._is_queue = bval