class Imu(object):
    def __init__(self):
        self.acc = None
        self.om = None

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def write(self, filepath):
        """ Writes IMU trajectory to file. """

        with open(filename, 'w+') as f:
            for i, t in enumerate(self.t):
                a_str = f"{self.ax[i]:.9f} {self.ay[i]:.9f} {self.az[i]:.9f} "
                g_str = f"{self.om.x[i]:.9f} {self.om.y[i]:.9f} {self.om.z[i]:.9f}"
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + '\n')