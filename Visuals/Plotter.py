from . import line_formats as lf
from .decorators import *

import numpy as np

class Plotter(object):
    def __init__(self):
        self.min_t = None
        self.max_t = None

    def plot(self):
        pass

    def _get_plot_rc(self, ai, num_rows):
        """ Returns current row and column for plotting. """
        col = math.floor(ai / num_rows)
        row = ai - col * num_rows

        return row, col

    def _get_ax(self, axes, ai, num_rows):
        row, col = self._get_plot_rc(ai, num_rows)
        a = axes[row][col]
        a.set_visible(True)
        return a

    def _init_axes(self, axes, num_rows, num_cols):
        if axes is None:
            fig, axes = plt.subplots(num_rows, num_cols)
            fig.tight_layout()

        for a in axes.flatten():
            a.set_visible(False)
            a.grid(True)
            a.ticklabel_format(useOffset=False)
            a.set_xlim(left=self.min_t, right=self.max_t)

        return axes

    def _plot_vals(self, ax, vals_dict: dict):
        for name, v in vals_dict.items():
            vals = v['vals']
            if vals != []:
                ax.plot(v['t'], vals, label=name)

    ### Ax postfix
    def _ax_postfix(self, a, label, *values):
        latex_label = self._get_latex_label(label)
        a.set_title(latex_label)
        self._adjust_y_range(a, *values)

    def _get_latex_label(self, label):
        """ Creates string in LaTeX math format. """

        if len(label) == 1:
            return '$' + label + '$'
        elif 'dof' in label:
            return 'dof$_' + label[-1] + '$'
        elif label[-1] == 'c':
            if 'deg' in label:
                return '$' + label[0] + '_{' + label[1:-5] + 'c}$'
            else:
                return '$' + label[0] + '_{' + label[1:] + '}$'
        elif 'notch' in label:
            if 'dd' in label:
                return '$\\theta_{n, dd}$'
            elif 'd' in label:
                return '$\\theta_{n, d}$'
            else:
                return '$\\theta_{n}$'
        else:
            return '$' + label[0] + '_' + label[1] + '$'

    def _adjust_y_range(self, a, *values):
        min_val, max_val = self._adjust_y_range_vals(min(*values),
                                max(*values))
        a.set_ylim(bottom=min_val, top=max_val)

    def _adjust_y_range_vals(self, min_val, max_val):
        range_val = max_val - min_val

        # display of very small values
        if range_val < 0.001:
            min_val = min_val - 0.1
            max_val = max_val + 0.1
        else:
            min_val = min_val - 0.2 * range_val
            max_val = max_val + 0.2 * range_val

        return min_val, max_val

    ### Fig postfix
    def _fig_postfix(self, filename, axes, offset, num_rows, config):
        """ Late setting of line styles, save figure. """
        self._set_line_styles(axes)
        self._put_legend_near_first_plot(axes, offset, num_rows)
        self._set_title(config)
        self.save(filename)

    def _put_legend_near_first_plot(self, axes, offset, num_rows):
        r0, c0 = self._get_plot_rc(offset, num_rows)
        axes[r0][c0].legend(bbox_to_anchor=(1., 1.2), loc='lower right')

    def _set_line_styles(self, axes):
        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_format(line)

    def _set_plot_line_format(self, line):
        label = line.get_label()
        if label == 'kf':
            self._apply_lf_dict(line, lf.kf)
        elif 'ref' in label:
            self._apply_lf_dict(line, lf.imuref)
        elif 'interpl' in label:
            self._apply_lf_dict(line, lf.interpl)
        else:
            self._apply_lf_dict(line, lf.default)

    def _apply_lf_dict(self, line, lf_dict):
        for k, v in lf_dict.items():
            if isinstance(v, str):
                v = '\'' + v + '\''
            else:
                v = str(v)
            eval(f'line.set_{k}(' + v + ')')

    def _set_title(self, config):
        MSE = config.dof_metric
        Kp = config.scale_process_noise
        Km = config.scale_meas_noise

        if MSE:
            st = plt.suptitle(f"DOF_MSE {MSE:.3f}  (Kp {Kp:.2E}, Km {Km:.2E})",
            fontsize=10)
        else:
            st = plt.suptitle('')

        # shift subplots down:
        st.set_y(0.95)
        fig = plt.gcf()
        fig.subplots_adjust(top=0.8)

    def save(self, filename):
        if filename:
            plt.savefig(filename, dpi=200)

class CameraPlot(Plotter):
    def __init__(self, camera):
        self.min_t      = camera.traj.t[0]
        self.max_t      = camera.traj.t[-1]
        self.camera = camera

        self.traj       = camera.traj
        self.jump_labels = ['x']

    @show_plot
    def plot(self, config, axes=None):
        labels      = self.traj.labels[1:]
        num_cols    = 2
        offset      = 1

        self._get_plot_objects(filename=None,
                        labels=labels,
                        num_cols=num_cols,
                        axes=axes,
                        config=config)

    @show_plot
    def plot_notch(self, config, axes=None):
        self.jump_labels = []
        labels_camera = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        labels_notch = ['notch', 'notch_d', 'notch_dd']
        labels = labels_camera + labels_notch
        num_cols = 3

        self._get_plot_objects(filename=None,
                        labels=labels,
                        num_cols=num_cols,
                        axes=axes,
                        config=config)

    @plot_loop
    def _get_plot_objects(self, label, **kwargs):
        print(label)
        if label.startswith('r'):
            rx = self.camera.r[0,:]
            ry = self.camera.r[1,:]
            rz = self.camera.r[2,:]

            objs = [self.traj]
            vals = [eval(f'{label}')]
            
            return objs, vals
        elif 'notch' not in label:
            objs = [self.traj]
            vals = [self.traj.__dict__[label]]
            return objs, vals
        else:
            return [None], [None]

class FilterPlot(Plotter):
    def __init__(self, traj, cam_traj, imu_ref):
        self.min_t = None
        self.max_t = None

        self.traj       = traj
        self.cam_traj   = cam_traj
        self.imu_ref    = imu_ref

        """ labels that start at row = 1 """
        self.jump_labels = ['x', 'vx', 'rx', 'dof1', 'dof4', 'xc', 'rx_degc']

    def _set_plot_times(self, config, t_end):
        self.min_t = config.min_t
        self.max_t = min(self.traj.t[-1], config.max_t, t_end)

    @show_plot
    def plot(self, config, t_end):
        self._set_plot_times(config, t_end)
        self._plot_imu_states(config)
        self._plot_camera_states(config)

    @show_plot
    def plot_compact(self, config, t_end):
        self._set_plot_times(config, t_end)

        self.jump_labels = []
        labels_cam = self.traj.labels_camera[:6]
        labels_imu = self.traj.labels_imu[3:9]
        labels_imu_dofs = self.traj.labels_imu_dofs

        labels = labels_cam + labels_imu + labels_imu_dofs

        num_cols = 6

        self._get_plot_objects(labels = labels,
                        labels_cam  = labels_cam,
                        labels_imu  = labels_imu,
                        labels_imu_dofs = labels_imu_dofs,
                        labels_notch = [],
                        config      = config,
                        num_cols    = num_cols,
                        filename    = config.img_filepath_compact,
                        cam         = self.cam_traj,
                        imu_ref     = self.imu_ref,
                        imu_recon   = None,
                        axes        = None,)

    def _plot_imu_states(self, config, imu_recon=None,
            axes=None):
        labels = self.traj.labels_imu + self.traj.labels_imu_dofs
        num_cols = 6

        self._get_plot_objects(labels = labels,
                        labels_cam  = [],
                        labels_imu  = self.traj.labels_imu,
                        labels_imu_dofs = self.traj.labels_imu_dofs,
                        labels_notch = [],
                        config      = config,
                        num_cols = num_cols,
                        filename = config.img_filepath_imu,
                        cam = None,
                        imu_ref = self.imu_ref,
                        imu_recon = imu_recon,
                        axes = axes)

    def _plot_camera_states(self, config, axes=None):
        self.jump_labels = []
        labels_camera = self.traj.labels_camera[:6]
        labels_notch = ['notch', 'notch_d', 'notch_dd']
        labels = labels_camera + labels_notch
        num_cols = 3

        self._get_plot_objects(labels = labels,
                        labels_cam  = labels_camera,
                        labels_imu  = [],
                        labels_imu_dofs = [],
                        labels_notch = labels_notch,
                        config      = config,
                        num_cols = num_cols,
                        filename = config.img_filepath_cam,
                        cam = self.cam_traj,
                        imu_ref = None,
                        imu_recon = None,
                        axes = axes)

    @plot_loop
    def _get_plot_objects(self, label, **kwargs):
        config      = kwargs['config']
        cam         = kwargs['cam']
        imu_ref     = kwargs['imu_ref']
        imu_recon   = kwargs['imu_recon']
        labels_cam  = kwargs['labels_cam']
        labels_imu  = kwargs['labels_imu']
        labels_imu_dofs  = kwargs['labels_imu_dofs']
        labels_notch  = kwargs['labels_notch']

        val_filt   = self.traj.__dict__[label] if label not in labels_notch else []

        if label in labels_cam:
            val_cam    = cam.__dict__[label[:-1]] \
                                if cam else []
        elif label in labels_notch:
            val_cam    = np.rad2deg(cam.__dict__[label]) \
                                if cam else []
        else:
            val_cam = []

        val_imuref = imu_ref.__dict__[label] if \
                        (imu_ref and label in labels_imu) \
                            else []
        val_recon  = imu_recon.__dict__[label] \
                        if (imu_recon and label in imu_recon.__dict__) \
                            else []

        if label in labels_imu_dofs:
            idx = labels_imu_dofs.index(label)
            dof_ref = DofRef(config.min_t, config.max_t)
            val_dof_ref = [config.real_imu_dofs[idx]] * 2
        else:
            dof_ref, val_dof_ref = None, []

        if cam:
            cam.name = 'ref'

        objs = [self.traj, cam, imu_ref, imu_recon, dof_ref]
        vals = [val_filt, val_cam, val_imuref, val_recon, val_dof_ref]

        return objs, vals

class DofRef(object):
    def __init__(self, min_t, max_t):
        self.name = 'dof ref'
        self.t = [min_t, max_t]
        self.vals = []