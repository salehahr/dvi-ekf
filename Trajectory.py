import os
import math
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from Measurement import VisualMeasurement, ImuMeasurement

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
            for label in self.labels:
                exec(f"self.{label} = []")

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

    def append_from_state(self, t, state):
        x, y, z = state.p
        qw, qx, qy, qz = quaternion.as_float_array(state.q)

        for label in self.labels:
            exec(f"self.{label}.append({label})")

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
            eval(f"axes[{row}][{col}].plot(self.t, self.{label}, label=self.name)")

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

class ImuTraj(Trajectory):
    def __init__(self, name="", filepath=None, vis_data=None, num_imu_between_frames=0):
        labels = ['t', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']

        self.name = name
        self.labels = labels
        self.filepath = filepath
        self.num_imu_between_frames = num_imu_between_frames

        if vis_data:
            self._init_from_visualtraj(vis_data)
        else:
            self._init_from_filepath()

        self.next_frame_index = 0
        self.queue_first_ts = 0

    def _init_from_filepath(self):
        super().__init__(self.name, self.labels, self.filepath)

    def _init_from_visualtraj(self, VisualTraj):
        self.generate_data(VisualTraj)

    def _get_next_frame_index(self, cam_t):
        """ Get index for which IMU time matches current camera time """
        return max([i for i, t in enumerate(self.t) if t <= cam_t])

    def get_queue(self, old_t, current_cam_t):
        start_index = self.next_frame_index
        prev_index = self._get_next_frame_index(old_t)
        next_index = self._get_next_frame_index(current_cam_t)
        self.next_frame_index = next_index + 1

        if (start_index <= prev_index) and \
            not (start_index == next_index):
            start_index = prev_index + 1

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
        queue.is_queue = True

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

    def generate_data(self, vis_data):
        self._gen_unnoised_imu(vis_data)
        self._gen_noised_imu()

    def _gen_unnoised_imu(self, vis_data):
        t = vis_data.t
        len_t = len(t)
        dt = t[1] - t[0]

        self.ax = self._get_acceleration_from_vpos(vis_data.x, dt)
        self.ay = self._get_acceleration_from_vpos(vis_data.y, dt)
        self.az = self._get_acceleration_from_vpos(vis_data.z, dt)

        rx, ry, rz = self._get_angles_from_vquats(vis_data, len_t)
        self.gx = np.gradient(rx, dt)
        self.gy = np.gradient(ry, dt)
        self.gz = np.gradient(rz, dt)

        self._interpolate_imu(vis_data.t, len_t)
        self._write_to_file()

    def _interpolate_imu(self, t, num_cam_datapoints):
        tmin = t[0]
        tmax = t[-1]

        num_imu_datapoints = (num_cam_datapoints - 1) * self.num_imu_between_frames + 1
        self.t = np.linspace(tmin, tmax, num=num_imu_datapoints)

        for label in self.labels:
            if label == 't':
                continue

            exec(f"f = interp1d(t, self.{label}, kind='linear')")
            exec(f"self.{label} = f(self.t)")

    def _gen_noised_imu(self):
        filename, ext = os.path.splitext(self.filepath)
        filename_noisy = filename + '_noisy' + ext

        cov_ax = 0.004
        cov_ay = 0.003
        cov_az = 0.002
        cov_gx = 0.07
        cov_gy = 0.005
        cov_gz = 0.07

        for label in self.labels:
            if label == 't':
                continue
            exec(f"self.{label} += np.random.normal(loc=0., scale=cov_{label}, size=len(self.t))")

        self._write_to_file(filename_noisy)

    def _get_acceleration_from_vpos(self, data, dt):
        diff = np.gradient(data, dt)
        return np.gradient(diff, dt)

    def _get_angles_from_vquats(self, data, len_t):
        rx = np.zeros((len_t,))
        ry = np.zeros((len_t,))
        rz = np.zeros((len_t,))

        for i in range(len_t):
            x = data.qx[i]
            y = data.qy[i]
            z = data.qz[i]
            w = data.qw[i]

            quat = np.quaternion(w, x, y, z)
            rx[i], ry[i], rz[i] = quaternion.as_euler_angles(quat)

        return rx, ry, rz

    def _write_to_file(self, filename=None):
        if filename == None:
            filename = self.filepath

        with open(filename, 'w+') as f:
            for i, t in enumerate(self.t):
                a_str = f"{self.ax[i]:.9f} {self.ay[i]:.9f} {self.az[i]:.9f} "
                g_str = f"{self.gx[i]:.9f} {self.gy[i]:.9f} {self.gz[i]:.9f}"
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + '\n')

    def plot(self, axes=None, min_t=None, max_t=None):
        axes = super().plot(axes, min_t, max_t)

        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_style(line)

        return axes

    def _set_plot_line_style(self, line):
        label = line.get_label()
        if 'noisy' in label:
            line.set_color('darkgrey')
            line.set_linewidth(0.2)
        elif 'gt' in label:
            line.set_color('black')
            line.set_linewidth(1)