import os
import math
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splrep, splev
from scipy.integrate import cumtrapz

from Measurement import VisualMeasurement, ImuMeasurement

class Trajectory(object):
    """ Base trajectory class which requires a
        trajectory name, trajectory labels, and
        a filepath.
    """

    def __init__(self, name, labels, filepath=None, cap=None):
        self.name = name
        self.labels = labels
        self.filepath = filepath
        self.cap = cap

        self.clear()
        if filepath:
            self._parse(cap)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def clear(self):
        """ Reinitialises data containers. """
        for label in self.labels:
            self.__dict__[label] = []

    def _parse(self, cap):
        """ Extract data from file."""

        with open(self.filepath, 'r') as f:
            for i, line in enumerate(f):
                data = line.split()

                # iterate over data containers
                for j, label in enumerate(self.labels):
                    meas = float(data[j])
                    self.__dict__[label].append(meas)

                if cap is not None:
                    if i >= cap - 1:
                        break

    def plot(self, axes=None, min_t=None, max_t=None):
        """ Creates a two column plot of the states/data. """

        num_labels = len(self.labels) - 1
        num_rows = math.ceil( num_labels / 2 )
        offset = num_labels % 2 # 0 if even, 1 if odd number of labels

        if axes is None:
            fig, axes = plt.subplots(num_rows, 2)
            fig.tight_layout()

        if offset == 1:
            axes[0,0].set_visible(False)

        ai = offset
        for i, label in enumerate(self.labels):
            # skip time data
            if label == 't':
                continue

            row, col = self._get_plot_rc(ai, num_rows)
            axes[row][col].plot(self.t, self.__dict__[label],
                label=self.name)

            latex_label = self._get_latex_label(label)
            axes[row][col].set_title(latex_label)
            axes[row][col].set_xlim(left=min_t, right=max_t)
            axes[row][col].grid(True)

            ai += 1

        # legend on last plot
        axes[row][col].legend()

        # late setting of line styles
        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_style(line)

        return axes

    def _get_plot_rc(self, ai, num_rows):
        """ Returns current row and column for plotting. """

        if ai <= (num_rows - 1):
            row = ai
            col = 0
        else:
            row = ai - num_rows
            col = 1

        return row, col

    def _get_latex_label(self, label):
        """ Creates string in LaTeX math format. """

        if len(label) == 1:
            return '$' + label + '$'
        else:
            return '$' + label[0] + '_' + label[1] + '$'

    def _set_plot_line_style(line):
        pass

class VisualTraj(Trajectory):
    """ Visual trajectory containing time and pose. """

    def __init__(self, name, filepath=None, cap=None):
        labels = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        super().__init__(name, labels, filepath, cap)

        self.interpolated = None
        self.reconstructed = None

    def at_index(self, index):
        """ Returns single visual measurement at the given index. """

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

    def append_state(self, t, state):
        """ Appends new measurement from current state. """

        x, y, z = state.p
        qw, qx, qy, qz = quaternion.as_float_array(state.q)
        data = np.hstack((t, x, y, z, qx, qy, qz, qw))

        for i, label in enumerate(self.labels):
            self.__dict__[label].append(data[i])

    def interpolate(self, num_imu_between_frames):
        """ Generates interpolated/fitted data points between frames. """

        interpolated = VisualTraj(self.name + ' interpl', self.filepath)

        tmin = self.t[0]
        tmax = self.t[-1]
        num_cam_datapoints = len(self.t)

        num_imu_datapoints = (num_cam_datapoints - 1) \
            * num_imu_between_frames + 1
        interpolated.t = np.linspace(tmin, tmax, num=num_imu_datapoints)

        for label in self.labels:
            if label == 't':
                continue

            val = self.__dict__[label]

            # interpolating
            f = splrep(self.t, val, k=5)
            interpolated.__dict__[label] = splev(interpolated.t, f)

        self.interpolated = interpolated

    def _set_plot_line_style(self, line):
        """ Defines line styles for IMU plot. """

        label = line.get_label()
        if 'interpl' in label:
            line.set_color('black')
            line.set_linewidth(1)
            line.set_linestyle(':')
            line.set_marker('o')
            line.set_markersize(2)
        elif 'recon'in label:
            line.set_linewidth(2)
            line.set_linestyle('--')
            line.set_color('tab:red')
        elif 'GT' in label or 'gt' in label:
            line.set_linewidth(1)
            line.set_color('tab:green')
        elif label == 'mono':
            line.set_linewidth(1)
            line.set_color('tab:orange')
        elif label == 'kf':
            line.set_linewidth(1)
            line.set_color('tab:blue')
        else:
            line.set_color('darkgrey')
            line.set_linewidth(0.5)

    def reconstruct_from_imu(self, data):
        """ Generates trajectory from IMU data using
        the available initial conditions. """

        reconstructed = VisualTraj(self.name + ' recon', self.filepath)

        t = data.t
        dt = t[1] - t[0]
        reconstructed.t = t
        
        # initial conditions
        x0, y0, z0 = self.x[0], self.y[0], self.z[0]
        vx0, vy0, vz0 = data.vx[0], data.vy[0], data.vz[0]
        qx0, qy0, qz0, qw0 = self.qx[0], self.qy[0], self.qz[0], self.qw[0]

        # iontegrating for pos
        vx = cumtrapz(data.ax, t, initial=0) + vx0
        vy = cumtrapz(data.ay, t, initial=0) + vy0
        vz = cumtrapz(data.az, t, initial=0) + vz0

        reconstructed.x = cumtrapz(vx, t, initial=0) + x0
        reconstructed.y = cumtrapz(vy, t, initial=0) + y0
        reconstructed.z = cumtrapz(vz, t, initial=0) + z0
        
        # integrating for orientation
        quat0 = np.quaternion(qw0, qx0, qy0, qz0)
        rx0, ry0, rz0 = quaternion.as_euler_angles(quat0)
        rx = cumtrapz(data.gx, t, initial=0) + rx0
        ry = cumtrapz(data.gy, t, initial=0) + ry0
        rz = cumtrapz(data.gz, t, initial=0) + rz0

        quats = quaternion.from_euler_angles(rx, ry, rz)
        quats_f = quaternion.as_float_array(quats)
        reconstructed.qw = quats_f[:,0]
        reconstructed.qx = quats_f[:,1]
        reconstructed.qy = quats_f[:,2]
        reconstructed.qz = quats_f[:,3]
        
        self.reconstructed = reconstructed
        
        return reconstructed

class ImuTraj(Trajectory):
    """ IMU trajectory containing the acceleration and
    angular velocity measurements. """

    def __init__(self, name="imu", filepath=None, vis_data=None, cap=None,
        num_imu_between_frames=0,
        covariance = [0.] * 6):

        labels = ['t', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        self.num_imu_between_frames = num_imu_between_frames
        self._flag_gen_unnoisy_imu = False
        self.noisy = None

        super().__init__(name, labels, filepath, cap)

        if vis_data:
            self.clear()
            self._gen_unnoisy_imu(vis_data)
            self._gen_noisy_imu(covariance)

        self.next_frame_index = 0
        self.queue_first_ts = 0

    def _gen_unnoisy_imu(self, vis_data):
        """ Generates IMU data from visual trajectory, without noise. """

        vis_data.interpolate(self.num_imu_between_frames)
        interpolated = vis_data.interpolated
        t = interpolated.t
        len_t = len(t)
        dt = t[1] - t[0]

        self.vx = np.gradient(interpolated.x, dt)
        self.vy = np.gradient(interpolated.y, dt)
        self.vz = np.gradient(interpolated.z, dt)

        self.ax = np.gradient(self.vx, dt)
        self.ay = np.gradient(self.vy, dt)
        self.az = np.gradient(self.vz, dt)

        rx, ry, rz = self._get_angles_from_vquats(interpolated, len_t)
        self.gx = np.gradient(rx, dt)
        self.gy = np.gradient(ry, dt)
        self.gz = np.gradient(rz, dt)

        self.t = interpolated.t

        self._write_to_file()
        self._flag_gen_unnoisy_imu = True

    def _gen_noisy_imu(self, covariance):
        """ Generates IMU data from visual trajectory, with noise. """

        assert(self._flag_gen_unnoisy_imu == True)

        filename, ext = os.path.splitext(self.filepath)
        filename_noisy = filename + '_noisy' + ext

        noisy = ImuTraj(name="noisy imu",
            filepath=filename_noisy,
            num_imu_between_frames=self.num_imu_between_frames,
            covariance=covariance)
        noisy.clear()

        for i, label in enumerate(self.labels):
            if label == 't':
                noisy.t = self.t
                continue

            noisy.__dict__[label] = self.__dict__[label] \
                + np.random.normal(loc=0., scale=covariance[i-1],
                    size=len(self.t))
                    
        noisy.vx = self.vx
        noisy.vy = self.vy
        noisy.vz = self.vz

        self.noisy = noisy
        self._write_to_file(filename_noisy)

    def _interpolate_imu(self, t):
        """ Generates IMU data points between frames. """

        tmin = t[0]
        tmax = t[-1]
        num_cam_datapoints = len(t)

        num_imu_datapoints = (num_cam_datapoints - 1) * self.num_imu_between_frames + 1
        self.t = np.linspace(tmin, tmax, num=num_imu_datapoints)

        for label in self.labels:
            if label == 't':
                continue

            f = interp1d(t, self.__dict__[label], kind='linear')
            self.__dict__[label] = f(self.t)

    def _get_angles_from_vquats(self, interpolated, len_t):
        """ Converts visual orientation quaternions to Euler angles. """

        quats_arr = np.asarray(
            [np.quaternion(w, interpolated.qx[i],
                interpolated.qy[i], interpolated.qz[i])
                for i, w in enumerate(interpolated.qw)])

        euler = quaternion.as_euler_angles(quats_arr)
        return euler[:,0], euler[:,1], euler[:,2]

    def _write_to_file(self, filename=None):
        """ Writes IMU trajectory to file. """

        if filename == None:
            filename = self.filepath

        with open(filename, 'w+') as f:
            for i, t in enumerate(self.t):
                a_str = f"{self.ax[i]:.9f} {self.ay[i]:.9f} {self.az[i]:.9f} "
                g_str = f"{self.gx[i]:.9f} {self.gy[i]:.9f} {self.gz[i]:.9f}"
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + '\n')

    def at_index(self, index):
        """ Returns single IMU measurement at the given index. """

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

    def get_queue(self, old_t, current_cam_t):
        """ Get IMU queue after old_t and up to current_cam_t. """

        start_index = self.next_frame_index

        prev_index = self._get_next_frame_index(old_t) # end of old imu queue
        next_index = self._get_next_frame_index(current_cam_t)
        assert(prev_index <= next_index)

        # update start index, if outdated (less than prev_index)
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

        self.next_frame_index = next_index + 1

        return ImuMeasurement(t, acc, om)

    def _get_next_frame_index(self, cam_t):
        """ Get index for which IMU time matches current camera time """
        return max([i for i, t in enumerate(self.t) if t <= cam_t])

    def _set_plot_line_style(self, line):
        """ Defines line styles for IMU plot. """

        label = line.get_label()
        if 'noisy' in label:
            line.set_color('darkgrey')
            line.set_linewidth(1)
        elif 'gt' in label:
            line.set_color('black')
            # line.set_linewidth(1)
            line.set_linestyle('')
            line.set_marker('o')
            line.set_markersize(0.5)