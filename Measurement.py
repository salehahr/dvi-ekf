import numpy as np
import quaternion

class Measurement(object):
    def __init__(self, t, v1, v2):
        self.t = t
        self.vec = np.hstack((v1, v2))

class VisualMeasurement(Measurement):
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
        
    def at_index(self, index):
        if self.t.ndim == 0:
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