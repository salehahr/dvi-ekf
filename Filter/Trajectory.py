import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import interp1d, splrep, splev
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R

from .Measurement import VisualMeasurement, ImuMeasurement
from .Quaternion import Quaternion

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
            try:
                self._parse(cap)
            except FileNotFoundError:
                file = open(filepath, 'w+')
                file.close()

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

        # late setting of line styles
        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_style(line)

        # legend on last plot
        axes[row][col].legend()

        return axes

    def plot_3d(self, ax=None):
        if ax is None:
            fig = plt.figure()
            fig.tight_layout()
            ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.x, self.y, self.z, label=self.name)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # late setting of line styles
        for line in ax.get_lines():
            self._set_plot_line_style(line)

        ax.legend()

        return ax

    def plot_velocities(self, axes=None, min_t=None, max_t=None):
        num_labels = 3
        num_rows = 3

        if axes is None:
            fig, axes = plt.subplots(num_rows, 1)
            fig.tight_layout()

        for row, label in enumerate(['vx', 'vy', 'vz']):
            if 'imu' in self.name:
                t = self.t
            elif 'kf' in self.name:
                t = self.t
            else:
                t = self.interpolated.t

            axes[row].plot(t, self.__dict__[label],
                label=self.name)

            latex_label = self._get_latex_label(label)
            axes[row].set_title(latex_label)
            axes[row].set_xlim(left=min_t, right=max_t)
            axes[row].grid(True)

        # late setting of line styles
        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_style(line)

        # legend on last plot
        axes[row].legend()

        return axes

    def plot_sens_noise(self, Rp, Rq, Q, axes=None, min_t=None, max_t=None):
        num_labels = 4
        num_rows = 4
        suptitle = f"Rp = {Rp};\nRq = {Rq}'\n Q = {Q}"
        figname = f"./img/Rp{Rp}_Rq{Rq}_Q{Q}.png"

        if axes is None:
            fig, axes = plt.subplots(num_rows, 1)
            fig.tight_layout()
            st = fig.suptitle(suptitle, fontsize="x-large")

            # shift subplots down:
            st.set_y(0.95)
            fig.subplots_adjust(top=0.75)

        t = self.t

        for row, label in enumerate(['x', 'z', 'qy', 'qw']):
            axes[row].plot(t, self.__dict__[label],
                label=self.name)

            latex_label = self._get_latex_label(label)
            axes[row].set_title(latex_label)
            axes[row].set_xlim(left=min_t, right=max_t)
            axes[row].grid(True)

        # late setting of line styles
        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_style(line)

        # legend on last plot
        axes[row].legend()

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

    def _get_index_at(self, T):
        """ Get index for which timestamp matches the argument T. """
        return max([i for i, t in enumerate(self.t) if t <= T])

class VisualTraj(Trajectory):
    """ Visual trajectory containing time and pose. """

    def __init__(self, name, filepath=None, cap=None):
        labels = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        super().__init__(name, labels, filepath, cap)

        self.interpolated = None
        self.quats = None

        if self.qx:
            self._gen_quats_farray()

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
        vx, vy, vz = state.v
        qx, qy, qz, qw = state.q.xyzw
        data = [t, x, y, z, qx, qy, qz, qw]

        for i, label in enumerate(self.labels):
            self.__dict__[label].append(data[i])

        for i, label in enumerate(['vx', 'vy', 'vz']):
            if label not in self.__dict__:
                self.__dict__[label] = [eval(label)]
            else:
                self.__dict__[label].append(eval(label))

    def append_data(self, t, data_labels, data):
        """ Appends new data not already belonging to the existing
            labels. """

        for i, label in enumerate(data_labels):
            if label not in self.__dict__:
                self.__dict__[label] = [data[i]]
            else:
                self.__dict__[label].append(data[i])

    def interpolate(self, num_imu_between_frames):
        """ Generates interpolated/fitted data points between frames. """

        interpolated = VisualTraj(self.name + ' interpl')

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
        self.interpolated._gen_quats_farray()

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
            line.set_linewidth(0.75)
            line.set_linestyle('-')
            line.set_color('blue')
        else:
            line.set_color('darkgrey')
            line.set_linewidth(0.5)

    def _gen_quats_farray(self):
        self.quats = [Quaternion(x=self.qx[i],
                        y=self.qy[i], z=self.qz[i], w=w)
                        for i, w in enumerate(self.qw)]

    def get_meas(self, old_t, current_t):
        """ Gets measurement, if any, after old_t up to current_t. """

        prev_index = self._get_index_at(old_t) # end of old imu queue
        next_index = self._get_index_at(current_t)
        assert(prev_index <= next_index)

        if prev_index == next_index:
            return None
        else:
            t = self.t[next_index]

            x = self.x[next_index]
            y = self.y[next_index]
            z = self.z[next_index]
            pos = np.vstack((x, y, z))

            qx = self.qx[next_index]
            qy = self.qy[next_index]
            qz = self.qz[next_index]
            qw = self.qw[next_index]
            rot = np.vstack((qx, qy, qz, qw))

            return VisualMeasurement(t, pos, rot)

class ImuTraj(Trajectory):
    """ IMU trajectory containing the acceleration and
    angular velocity measurements. """

    def __init__(self, name="imu", filepath=None, vis_data=None,
        cap=None,
        num_imu_between_frames=0,
        covariance = [0.] * 6,
        unnoised = False):

        # base properties
        labels = ['t', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        super().__init__(name, labels, filepath, cap)
        self.vis_data = vis_data
        self.num_imu_between_frames = num_imu_between_frames

        # extra data
        self.noisy = None
        self.reconstructed = None

        # flags
        self._flag_gen_unnoisy_imu = False

        # data generation
        if vis_data:
            self.clear()
            self._gen_unnoisy_imu()

            if not unnoised:
                self._gen_noisy_imu(covariance)

        self.next_frame_index = 0
        self.queue_first_ts = 0

    def _gen_unnoisy_imu(self):
        """ Generates IMU data from visual trajectory, without noise.
            Involves transformation from SLAM world coordinates
            to IMU coordinates."""

        self.vis_data.interpolate(self.num_imu_between_frames)
        interpolated = self.vis_data.interpolated
        dt = interpolated.t[1] - interpolated.t[0]

        # get rotations for coordinate transformation
        R_BW_arr = [q.rot.T for q in interpolated.quats]

        # velocities
        v_W = np.asarray((np.gradient(interpolated.x, dt),
                        np.gradient(interpolated.y, dt),
                        np.gradient(interpolated.z, dt) )).T
        self.vis_data.vx = v_W[:,0] # VisTraj is in world coordinates
        self.vis_data.vy = v_W[:,1]
        self.vis_data.vz = v_W[:,2]

        v_B = np.asarray([R_BW @ v_W[i,:]
                for i, R_BW in enumerate(R_BW_arr)]) ## possibly wrong transformation?
        self.vx, self.vy, self.vz = v_B[:,0], v_B[:,1], v_B[:,2]

        # accelerations
        a_W = np.array((np.gradient(v_W[:,0], dt),
                        np.gradient(v_W[:,1], dt),
                        np.gradient(v_W[:,2], dt))).T

        a_B = np.asarray([R_BW @ a_W[i,:] for i, R_BW in enumerate(R_BW_arr)])
        self.ax, self.ay, self.az = a_B[:,0], a_B[:,1], a_B[:,2]

        # angular velocity
        euler = np.asarray([q.euler_zyx for q in interpolated.quats])
        rz, ry, rx = euler[:,0], euler[:,1], euler[:,2]

        om_W = np.asarray( (np.gradient(rx, dt),
                            np.gradient(ry, dt),
                            np.gradient(rz, dt)) )
        om_B = np.asarray([R_BW @ om_W[:,i] for i, R_BW in enumerate(R_BW_arr)]).T
        self.gx, self.gy, self.gz = om_B[0,:], om_B[1,:], om_B[2,:]

        self.t = interpolated.t

        if self.filepath:
            self._write_to_file()

        self._flag_gen_unnoisy_imu = True

    def _gen_noisy_imu(self, covariance):
        """ Generates IMU data from visual trajectory, with noise. """

        assert(self._flag_gen_unnoisy_imu == True)

        if self.filepath:
            filename, ext = os.path.splitext(self.filepath)
            filename_noisy = filename + '_noisy' + ext
        else:
            filename_noisy = None

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

        self.noisy = noisy

        if self.filepath:
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

        prev_index = self._get_index_at(old_t) # end of old imu queue
        next_index = self._get_index_at(current_cam_t)
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

    def reconstruct_vis_traj(self):
        """ Generates trajectory from IMU data using
        the available initial conditions. """

        reconstructed = VisualTraj('recon')

        t = self.t
        dt = t[1] - t[0]
        reconstructed.t = t

        # get rotations for coordinate transformation
        R_WB_arr = [q.rot for q in self.vis_data.interpolated.quats]

        # initial conditions in world coordinates
        R_WB0 = R_WB_arr[0]
        x0, y0, z0 = self.vis_data.x[0], self.vis_data.y[0], self.vis_data.z[0]
        v0 = self._to_world_coords(R_WB0, np.asarray((self.vx[0],
                                            self.vy[0],
                                            self.vz[0])))
        q0 = self.vis_data.quats[0]

        # integrating for pos
        a_W = self._to_world_coords(R_WB_arr, np.asarray((self.ax,
                                            self.ay,
                                            self.az)).T )
        vx_W = cumtrapz(a_W[:,0], t, initial=0) + v0[0]
        vy_W = cumtrapz(a_W[:,1], t, initial=0) + v0[1]
        vz_W = cumtrapz(a_W[:,2], t, initial=0) + v0[2]

        reconstructed.x = cumtrapz(vx_W, t, initial=0) + x0
        reconstructed.y = cumtrapz(vy_W, t, initial=0) + y0
        reconstructed.z = cumtrapz(vz_W, t, initial=0) + z0

        # integrating for orientation
        rz0, ry0, rx0 = q0.euler_zyx

        om_B = np.array((self.gx, self.gy, self.gz))
        om_W = np.asarray([R_WB @ om_B[:,i] for i, R_WB in enumerate(R_WB_arr)]).T

        rx = cumtrapz(om_W[0,:], t, initial=0) + rx0
        ry = cumtrapz(om_W[1,:], t, initial=0) + ry0
        rz = cumtrapz(om_W[2,:], t, initial=0) + rz0

        euler_ang = np.asarray([rz, ry, rx]).T
        quats = np.asarray([R.from_euler('zyx', e).as_quat()
            for e in euler_ang])
        reconstructed.qx = quats[:,0]
        reconstructed.qy = quats[:,1]
        reconstructed.qz = quats[:,2]
        reconstructed.qw = quats[:,3]

        self.reconstructed = reconstructed

        return reconstructed

    def _set_plot_line_style(self, line):
        """ Defines line styles for IMU plot. """

        line.set_linestyle('')
        line.set_marker('o')
        line.set_markersize(0.4)

        label = line.get_label()
        if 'noisy' in label:
            line.set_linestyle('-')
            line.set_color('darkgrey')
            line.set_linewidth(1)
        elif 'gt' in label:
            line.set_color('black')
        elif 'mono' in label:
            line.set_color('saddlebrown')

    def _to_world_coords(self, R_WB, vec):
        if isinstance(R_WB, list):
            assert(vec.shape[1] == 3)
            return np.asarray([Rot @ vec[i,:] for i, Rot in enumerate(R_WB)])
        else:
            return R_WB @ vec

    def _to_imu_coords(self, R_BW, vec):
        self._to_world_coords(R_BW, vec)