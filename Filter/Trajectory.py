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

from spatialmath import SE3

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

        self._nvals = 0
        self._flag_interpolated = False
        self._num_imu_between_frames = 1

        self.clear()
        if filepath:
            try:
                self._parse(cap)
            except FileNotFoundError:
                file = open(filepath, 'w+')
                file.close()

    @property
    def nvals(self):
        return len(self.t)

    @property
    def flag_interpolated(self):
        return self._flag_interpolated

    @property
    def num_imu_between_frames(self):
        return self._num_imu_between_frames

    @num_imu_between_frames.setter
    def num_imu_between_frames(self, val):
        self._num_imu_between_frames = val

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

    def interpolate(self, old_t, num_imu_between_frames, interp_obj=None):
        """ Generates data points between frames. """
        interp_obj = self if (interp_obj is None) else interp_obj
        interp_obj.num_imu_between_frames = num_imu_between_frames

        tmin = old_t[0]
        tmax = old_t[-1]
        num_old_datapoints = len(old_t)

        num_new_datapoints = (num_old_datapoints - 1) * num_imu_between_frames + 1
        new_t = np.linspace(tmin, tmax, num=num_new_datapoints)

        print(f"Interpolating {self.name} data: {num_old_datapoints} --> {num_new_datapoints} values.")

        for label in self.labels:
            if label == 't':
                interp_obj.t = new_t
                continue

            val = self.__dict__[label]

            # interpolating
            f = splrep(old_t, val, k=5)
            interp_obj.__dict__[label] = splev(new_t, f)

        interp_obj._flag_interpolated = True

    def plot(self, axes=None, min_t=None, max_t=None, dist=None,
        plot_euler=False, euler_extrinsic=True):
        """ Creates a two column plot of the states/data. """

        num_labels = len(self.labels) - 1
        num_cols = 2
        num_rows = math.ceil( num_labels / num_cols )
        offset = num_labels % 2 # 0 if even, 1 if odd number of labels

        if axes is None:
            fig, axes = plt.subplots(num_rows, num_cols)
            fig.tight_layout()

        if offset == 1 and dist is None:
            axes[0,0].set_visible(False)
        elif dist:
            axes[0,0].set_visible(True)
            axes[0,0].plot(self.t, dist, label=self.name)
            axes[0,0].set_title('Euclidian distance B-C')

        if plot_euler and offset == 1:
            axes[0,1].set_visible(False)

        ai = offset
        if plot_euler:
            plot_labels = ['t', 'x', 'y', 'z', 'qw', 'rx', 'ry', 'rz'] if euler_extrinsic else ['t', 'x', 'y', 'z', 'qw', 'rX', 'rY', 'rZ']
        else:
            plot_labels = self.labels

        for i, label in enumerate(plot_labels):
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

    def plot_allrots(self, axes=None, min_t=None, max_t=None, dist=None):
        """ Creates a two column plot of the states/data. """

        num_labels = len(self.labels) - 1
        num_cols = 3
        num_rows = math.ceil( num_labels / num_cols )
        offset = num_labels % 2 # 0 if even, 1 if odd number of labels

        if axes is None:
            fig, axes = plt.subplots(num_rows, num_cols)
            fig.tight_layout()

        if offset == 1 and dist is None:
            axes[0,0].set_visible(False)
        elif dist:
            axes[0,0].set_visible(True)
            axes[0,0].plot(self.t, dist, label=self.name)
            axes[0,0].set_title('Euclidian distance B-C')

        if offset == 1:
            axes[0,1].set_visible(False)

        ai = offset
        plot_labels = ['t', 'x', 'y', 'z', 'qw', 'rx', 'ry', 'rz', 'qw', 'qx', 'qy', 'qz']

        for i, label in enumerate(plot_labels):
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

    def plot_pc(self, axes=None, min_t=None, max_t=None):
        num_labels = 3
        num_rows = 3

        if axes is None:
            fig, axes = plt.subplots(num_rows, 1)
            fig.tight_layout()

        for row, label in enumerate(['cx', 'cy', 'cz']):
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
        col = math.floor(ai / num_rows)
        row = ai - col * num_rows

        return row, col

    def _get_latex_label(self, label):
        """ Creates string in LaTeX math format. """

        if len(label) == 1:
            return '$' + label + '$'
        elif 'dof' in label:
            return 'dof$_' + label[-1] + '$'
        elif label[-1] == 'c':
            return '$' + label[0] + '_{' + label[1:] + '}$'
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
            self.gen_angle_arrays()

    @property
    def R(self):
        return [q.rot for q in self.quats]

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

        self.interpolated = VisualTraj(self.name + ' interpl')
        super().interpolate(self.t, num_imu_between_frames, self.interpolated)
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

    def gen_angle_arrays(self):
        self._gen_quats_farray()
        self._gen_euler_angles()

    def _gen_quats_farray(self):
        self.quats = [Quaternion(x=self.qx[i],
                        y=self.qy[i], z=self.qz[i], w=w)
                        for i, w in enumerate(self.qw)]

    def _gen_euler_angles(self):
        """ extrinsic: xyz: rotations about fixed CS
            intrinsic: XYZ: rotations about moving CS """

        euler = np.array([R.from_quat(q.xyzw).as_euler('xyz', degrees=True) for q in self.quats])
        self.rx = euler[:,0]
        self.ry = euler[:,1]
        self.rz = euler[:,2]

        euler = np.array([R.from_quat(q.xyzw).as_euler('XYZ', degrees=True) for q in self.quats])
        self.rX = euler[:,0]
        self.rY = euler[:,1]
        self.rZ = euler[:,2]

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

class ImuDesTraj(Trajectory):
    """ Desired traj of the IMU. """

    def __init__(self, name, imu):
        labels = ['t', 'x', 'y', 'z',
                    'vx', 'vy', 'vz',
                    'rx', 'ry', 'rz',
                    'qw', 'qx', 'qy', 'qz']
        super().__init__(name, labels)
        self.imu = imu

    def append_value(self, t, current_cam):
        """ Appends new measurement from current state. """

        p, R_WB, v = self.imu.desired_vals(current_cam)

        euler_angs = R.from_matrix(R_WB).as_euler('xyz', degrees=True)
        quats = Quaternion(val=R_WB)
        data = [t, *p, *v, *euler_angs, *quats.wxyz]

        for i, label in enumerate(self.labels):
            self.__dict__[label].append(data[i])

    def from_cam(self, cam):
        queue = cam.generate_queue(cam.t[0], cam.t[-1])
        for n, t in enumerate(queue.t):
            current_cam = queue.at_index(n)
            self.append_value(t, current_cam)

class FilterTraj(Trajectory):
    def __init__(self, name):
        self.labels_imu = ['x', 'y', 'z',
                    'vx', 'vy', 'vz',
                    'rx', 'ry', 'rz',
                    'qw', 'qx', 'qy', 'qz']
        self.labels_imu_dofs = [ 'dof1', 'dof2', 'dof3',
                    'dof4', 'dof5', 'dof6']
        self.labels_camera = ['xc', 'yc', 'zc',
                    'rxc', 'ryc', 'rzc',
                    'qwc', 'qxc', 'qyc', 'qzc']
        labels = ['t', *self.labels_imu,
                    *self.labels_imu_dofs, *self.labels_camera]
        super().__init__(name, labels)

    def append_state(self, t, state):
        """ Appends new measurement from current state. """

        euler_angs = R.from_quat(state.q.xyzw).as_euler('xyz', degrees=True)
        euler_angs_C = R.from_quat(state.q_cam.xyzw).as_euler('xyz', degrees=True)
        data = [t, *state.p, *state.v, *euler_angs, *state.q.wxyz,
                    *state.dofs,
                    *state.p_cam, *euler_angs_C, *state.q_cam.wxyz]

        for i, label in enumerate(self.labels):
            # print(f'{i} {label}: {data[i]}')
            self.__dict__[label].append(data[i])

    def plot(self, labels, num_cols, offset, filename='',
            cam=None, imu_ref=None, axes=None, min_t=None, max_t=None):
        num_labels = len(labels)
        num_rows = math.ceil( num_labels / num_cols )

        if axes is None:
            fig, axes = plt.subplots(num_rows, num_cols)
            fig.tight_layout()

        for row in range(num_rows):
            for col in range(num_cols):
                axes[row,col].set_visible(False)

        ai = offset
        for i, label in enumerate(labels):
            if label in ['vx', 'rx', 'dof1', 'dof4', 'rxc']:
                ai += 1

            row, col = self._get_plot_rc(ai, num_rows)

            val_filt = self.__dict__[label]
            val_cam = cam.__dict__[label[:-1]] if cam else []
            val_des = imu_ref.__dict__[label] if (imu_ref and 'dof' not in label) else []

            min_val, max_val = min(*val_filt, *val_cam, *val_des), max(*val_filt, *val_cam, *val_des)

            range_val = max_val - min_val

            # display of very small values
            if range_val < 0.01:
                min_val = min_val - 1
                max_val = max_val + 1
            else:
                min_val = min_val - 0.2 * range_val
                max_val = max_val + 0.2 * range_val

            axes[row][col].set_visible(True)
            if val_filt:
                axes[row][col].plot(self.t, val_filt, label=self.name)
            if val_cam != []:
                axes[row][col].plot(cam.t, val_cam, label=cam.name)
            if val_des != []:
                axes[row][col].plot(imu_ref.t, val_des, label=imu_ref.name)

            latex_label = self._get_latex_label(label)
            axes[row][col].set_title(latex_label)
            axes[row][col].set_xlim(left=min_t, right=max_t)
            axes[row][col].set_ylim(bottom=min_val, top=max_val)
            axes[row][col].grid(True)

            ai += 1

        # late setting of line styles
        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_style(line)

        # legend on first plot
        r0, c0 = self._get_plot_rc(offset, num_rows)
        axes[r0][c0].legend(bbox_to_anchor=(1., 2.))

        # save img
        if filename:
            plt.savefig(filename, dpi=200)

        return axes

    def plot_imu(self, filename='', imu_ref=None, axes=None, min_t=None, max_t=None):
        """ Creates plot of the IMU positioning parameters on the probe. """

        labels = self.labels_imu + self.labels_imu_dofs
        num_cols = 6
        offset = len(labels) % 2
        return self.plot(labels, num_cols, offset, imu_ref=imu_ref, filename=filename, axes=axes, min_t=min_t, max_t=max_t)

    def plot_dofs(self, axes=None, min_t=None, max_t=None):
        """ Creates plot of the IMU positioning parameters on the probe. """

        labels = self.labels_imu_dofs

        num_labels = len(labels)
        num_cols = 2
        num_rows = math.ceil( num_labels / num_cols )

        if axes is None:
            fig, axes = plt.subplots(num_rows, num_cols)
            fig.tight_layout()

        for i, label in enumerate(labels):
            row, col = self._get_plot_rc(i, num_rows)

            val = self.__dict__[label]
            min_val, max_val = min(val), max(val)

            # display of very small values
            if max_val - min_val < 0.001:
                min_val = min_val - 1
                max_val = max_val + 1

            axes[row][col].plot(self.t, val,
                label=self.name)

            latex_label = self._get_latex_label(label)
            axes[row][col].set_title(latex_label)
            axes[row][col].set_xlim(left=min_t, right=max_t)
            axes[row][col].set_ylim(bottom=min_val, top=max_val)
            axes[row][col].grid(True)

        # late setting of line styles
        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_style(line)

        # legend on last plot
        axes[row][col].legend()

        return axes

    def plot_camera(self, filename, cam, axes=None, min_t=None, max_t=None):
        """ Creates plot of the camera states/data. """

        labels = self.labels_camera
        num_cols = 3
        offset = 1
        return self.plot(labels, num_cols, offset, cam=cam, filename=filename, axes=axes, min_t=min_t, max_t=max_t)

    def plot_3d(self, cam=None, imu_ref=None, ax=None):
        if ax is None:
            fig = plt.figure()
            fig.tight_layout()
            ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.x, self.y, self.z, label='imu est')
        if imu_ref:
            ax.plot(imu_ref.x, imu_ref.y, imu_ref.z, label='imu ref')

            B = np.eye(4)
            B[:3, :3] = Quaternion(w=imu_ref.qw[0],
                    v=np.array([imu_ref.qx[0], imu_ref.qy[0], imu_ref.qz[0]]),
                    do_normalise=True).rot
            B[:3, -1] = [imu_ref.x[0], imu_ref.y[0], imu_ref.z[0]]
            X = SE3(B)
            X.plot(frame='B', arrow=False, axes=fig.axes[0], length=0.1, color='tab:green')

        ax.plot(self.xc, self.yc, self.zc, label='cam est')
        if cam:
            ax.plot(cam.x, cam.y, cam.z, label='cam ref')

            C = np.eye(4)
            C[:3, :3] = cam.quats[0].rot
            C[:3, -1] = [cam.x[0], cam.y[0], cam.z[0]]
            X = SE3(C)
            X.plot(frame='C', arrow=False, axes=fig.axes[0], length=0.1, color='tab:green')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # late setting of line styles
        for line in ax.get_lines():
            self._set_plot_line_style(line)

        ax.legend()
        plt.legend()
        plt.show()

        return ax

    def _set_plot_line_style(self, line):
        """ Defines line styles for IMU plot. """

        label = line.get_label()
        if label == 'kf':
            line.set_linewidth(0.75)
            line.set_linestyle('-')
            line.set_color('blue')
        elif 'ref' in label:
            line.set_linewidth(0.75)
            line.set_linestyle('--')
            line.set_color('tab:green')
        elif 'est' in label:
            line.set_linewidth(0.75)
            line.set_linestyle('-')
            line.set_color('blue')
        else:
            line.set_color('darkgrey')
            line.set_linestyle('--')
            line.set_linewidth(0.8)

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

    def interpolate(self):
        """ Generates IMU data points between frames. """
        super().interpolate(self.t, self.num_imu_between_frames)

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

    def reconstruct(self, R_WB, W_p_BW_0, WW_v_BW_0):
        """ For validation.
            Generates trajectory from IMU data.
            The IMU trajectory is obtained via numerical integration
            using the available initial conditions. """

        reconstructed = VisualTraj('recon')

        t = self.t
        dt = t[1] - t[0]
        reconstructed.t = t

        # initial conditions in world coordinates
        x0, y0, z0 = W_p_BW_0
        vx0, vy0, vz0 = WW_v_BW_0
        rz0, ry0, rx0 = Quaternion(val=R_WB[0], do_normalise=True).euler_zyx

        # velocity in world coordinates
        assert(len(R_WB) == len(self.ax))
        W_acc = self._to_world_coords(R_WB,
                    np.asarray((self.ax,
                                self.ay,
                                self.az)).T )

        W_vx = cumtrapz(W_acc[:,0], t, initial=0) + vx0
        W_vy = cumtrapz(W_acc[:,1], t, initial=0) + vy0
        W_vz = cumtrapz(W_acc[:,2], t, initial=0) + vz0

        # position in world coordinates
        reconstructed.x = cumtrapz(W_vx, t, initial=0) + x0
        reconstructed.y = cumtrapz(W_vy, t, initial=0) + y0
        reconstructed.z = cumtrapz(W_vz, t, initial=0) + z0

        # orientation in world
        W_om_B = self._to_world_coords(R_WB,
                    np.asarray((self.gx,
                                self.gy,
                                self.gz)).T )

        rx = cumtrapz(W_om_B[:,0], t, initial=0) + rx0
        ry = cumtrapz(W_om_B[:,1], t, initial=0) + ry0
        rz = cumtrapz(W_om_B[:,2], t, initial=0) + rz0

        euler_ang = np.asarray([rz, ry, rx]).T
        rots = [R.from_euler('zyx', e).as_matrix()
            for e in euler_ang]
        quats = [Quaternion(val=R_i, do_normalise=True) for R_i in rots]

        reconstructed.qx = [q.x for q in quats]
        reconstructed.qy = [q.y for q in quats]
        reconstructed.qz = [q.z for q in quats]
        reconstructed.qw = [q.w for q in quats]
        reconstructed.gen_angle_arrays()

        self.reconstructed = reconstructed

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