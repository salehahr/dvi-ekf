import math
import numpy as np
from Measurement import VisualMeasurement, ImuMeasurement
import matplotlib.pyplot as plt

class Trajectory(object):
    def __init__(self, name, labels, filepath=None):
        self.name = name
        self.labels = labels
        self.filepath = filepath

        if filepath:
            data = self.parse(self.filepath, self.labels)
            for label in self.labels:
                exec(f"self.{label} = data[\'{label}\']")
        else:
            self.t = []
            self.x = []
            self.y = []
            self.z = []
            self.qx = []
            self.qy = []
            self.qz = []
            self.qw = []

    def parse(self, filepath, data_labels):
        data_containers = {}
        num_labels = len(data_labels)

        for label in data_labels:
            data_containers[label] = []

        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                data = line.split()
                ts = float(data[0])

                for j, label in enumerate(data_labels):
                    if label == 't':
                        data_containers['t'].append(ts)
                        continue

                    if j == (num_labels - 1):
                        meas = float(data[j].rstrip())
                    else:
                        meas = float(data[j])

                    data_containers[label].append(meas)

        # Convert list to numpy array
        for label in data_labels:
            data_containers[label] = np.asarray(data_containers[label])

        return data_containers

    def append_data(self, t, data):
        self.t.append(t)

        self.x.append(data[0])
        self.y.append(data[1])
        self.z.append(data[2])

        self.qx.append(data[3])
        self.qy.append(data[4])
        self.qz.append(data[5])
        self.qw.append(data[6])

    def plot(self, axes=None, min_t=None, max_t=None):
        num_labels = len(self.labels) - 1
        num_rows = math.ceil( num_labels / 2 )
        offset = num_labels % 2 # 0 if even, 1 if odd number of labels

        if axes is None:
            fig, axes = plt.subplots(num_rows, 2)
            fig.tight_layout()

        for i, label in enumerate(self.labels):
            # skip time data
            if label == 't':
                ai = i + offset
                continue                

            row, col = self._get_plot_rc(ai, num_rows)
            exec(f"{label} = axes[{row}][{col}].plot(self.t, self.{label}, label=self.name)")

            latex_label = self._get_latex_label(label)
            axes[row][col].set_title(latex_label)
            axes[row][col].set_xlim(left=min_t, right=max_t)
            axes[row][col].grid(True)

            ai += 1

        # legend on last plot
        axes[row][col].legend()

        return axes

    def _get_plot_rc(self, ai, num_rows):
        if ai <= (num_rows - 1):
            row = ai
            col = 0
        else:
            row = ai - num_rows
            col = 1

        return row, col

    def _get_latex_label(self, label):
        if len(label) == 1:
            return '$' + label + '$'
        else:
            return '$' + label[0] + '_' + label[1] + '$'

class VisualTraj(Trajectory):
    def __init__(self, name, filepath):
        labels = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        super().__init__(name, labels, filepath)

    def at_index(self, index):
        t = self.t[index]

        x = self.x[index]
        y = self.y[index]
        z = self.z[index]        
        pos = np.array([x, y, z])

        qx = self.qx[index]
        qy = self.qy[index]
        qz = self.qz[index]
        qw = self.qw[index]
        rot = np.array([qx, qy, qz, qw])

        return VisualMeasurement(t, pos, rot)

class ImuTraj(object):
    def __init__(self, name, filepath):
        labels = ['t', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        super().__init__(name, labels, filepath)

        self.next_frame_index = 0
        self.queue_first_ts = 0

    def get_next_frame_index(self, cam_t):
        return max([i for i, t in enumerate(self.t) if t <= cam_t])

    def get_imu_queue(self, cam_t):
        start_index = self.next_frame_index
        next_index = self.get_next_frame_index(cam_t)

        if start_index == next_index:
            queue = self.at_index(start_index)
        else:
            t = self.t[start_index:next_index+1]

            ax = self.ax[start_index:next_index+1]
            ay = self.ay[start_index:next_index+1]
            az = self.az[start_index:next_index+1]
            acc = np.vstack((ax, ay, az))

            gx = self.gx[start_index:next_index+1]
            gy = self.gy[start_index:next_index+1]
            gz = self.gz[start_index:next_index+1]
            om = np.vstack((gx, gy, gz))

            queue = ImuMeasurement(t, acc, om)

        self.next_frame_index = next_index + 1
        return queue

    def at_index(self, index):
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