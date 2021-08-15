from . import line_formats as lf
from .decorators import *

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
            return '$' + label[0] + '_{' + label[1:] + '}$'
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
    def _fig_postfix(self, filename, axes, offset, num_rows):
        """ Late setting of line styles, save figure. """
        self._set_line_styles(axes)
        self._put_legend_near_first_plot(axes, offset, num_rows)
        self.save(filename)

    def _put_legend_near_first_plot(self, axes, offset, num_rows):
        r0, c0 = self._get_plot_rc(offset, num_rows)
        axes[r0][c0].legend(bbox_to_anchor=(1., 2.))

    def _set_line_styles(self, axes):
        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_format(line)

    def _set_plot_line_format(self, line):
        label = line.get_label()
        if label == 'kf':
            self._apply_lf_dict(line, lf.kf)
        elif label == 'imu ref':
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

    def save(self, filename):
        if filename:
            plt.savefig(filename, dpi=200)

class CameraPlot(Plotter):
    def __init__(self, camera):
        self.min_t      = camera.traj.t[0]
        self.max_t      = camera.traj.t[-1]

        self.traj       = camera.traj
        self.jump_labels = ['x']

    @show_plot
    def plot(self, axes=None):
        labels      = self.traj.labels[1:]
        num_cols    = 2
        offset      = 1

        self._get_plot_objects(filename=None, labels=labels,
                        num_cols=num_cols, axes=axes)

    @plot_loop
    def _get_plot_objects(self, label, **kwargs):
        objs = [self.traj]
        vals = [self.traj.__dict__[label]]
        return objs, vals

class FilterPlot(Plotter):
    def __init__(self, traj, cam_traj, imu_ref):
        self.min_t = None
        self.max_t = None

        self.traj       = traj
        self.cam_traj   = cam_traj
        self.imu_ref    = imu_ref

        """ labels that start at row = 1 """
        self.jump_labels = ['x', 'vx', 'rx', 'dof1', 'dof4', 'xc', 'rxc']

    @show_plot
    def plot(self, config, t_end):
        self.min_t = config.min_t
        self.max_t = min(self.traj.t[-1], config.max_t, t_end)

        self._plot_imu_states(config)
        self._plot_camera_states(config)

    def _plot_imu_states(self, config, imu_recon=None,
            axes=None):
        labels = self.traj.labels_imu + self.traj.labels_imu_dofs
        num_cols = 6

        self._get_plot_objects(labels = labels,
                        num_cols = num_cols,
                        filename = config.img_filepath_imu,
                        cam = None,
                        imu_ref = self.imu_ref,
                        imu_recon = imu_recon,
                        axes = axes)

    def _plot_camera_states(self, config, axes=None):
        labels = self.traj.labels_camera
        num_cols = 3

        self._get_plot_objects(labels = labels,
                        num_cols = num_cols,
                        filename = config.img_filepath_cam,
                        cam = self.cam_traj,
                        imu_ref = None,
                        imu_recon = None,
                        axes = axes)

    @plot_loop
    def _get_plot_objects(self, label, **kwargs):
        cam         = kwargs['cam']
        imu_ref     = kwargs['imu_ref']
        imu_recon   = kwargs['imu_recon']

        val_filt   = self.traj.__dict__[label]
        val_cam    = cam.__dict__[label[:-1]] if cam \
                            else []
        val_imuref = imu_ref.__dict__[label] if \
                        (imu_ref and 'dof' not in label) \
                            else []
        val_recon  = imu_recon.__dict__[label] \
                        if (imu_recon and label in imu_recon.__dict__) \
                            else []

        objs = [self.traj, cam, imu_ref, imu_recon]
        vals = [val_filt, val_cam, val_imuref, val_recon]

        return objs, vals

