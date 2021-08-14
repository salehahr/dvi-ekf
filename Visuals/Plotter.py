import math
import matplotlib.pyplot as plt

default_line_format = { 'linewidth' : 0.8,
                        'linestyle' : '--',
                        'color'     : 'darkgrey'}

kf_line_format      = { 'linewidth' : 0.75,
                        'linestyle' : '-',
                        'color'     : 'blue'}

imuref_line_format  = { 'linewidth' : 0.75,
                        'linestyle' : '--',
                        'color'     : 'tab:green'}

def show_plot(plot_func, *args):
    def wrapper(*args):
        plot_func(*args)
        plt.show()
    return wrapper

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

    def _init_figure(self, axes, num_rows, num_cols):
        if axes is None:
            fig, axes = plt.subplots(num_rows, num_cols)
            fig.tight_layout()

        for a in axes.flatten():
            a.set_visible(False)
            a.grid(True)
            a.ticklabel_format(useOffset=False)
            a.set_xlim(left=self.min_t, right=self.max_t)

        return axes

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
    def _set_line_styles(self, axes):
        for ax in axes.reshape(-1):
            for line in ax.get_lines():
                self._set_plot_line_format(line)

    def _set_plot_line_format(self, line):
        pass

    def _apply_lf_dict(self, line, ls_dict):
        for k, v in ls_dict.items():
            if isinstance(v, str):
                v = '\'' + v + '\''
            else:
                v = str(v)
            eval(f'line.set_{k}(' + v + ')')

    def save(self, filename):
        if filename:
            plt.savefig(filename, dpi=200)

class FilterPlot(Plotter):
    def __init__(self, traj, cam_traj, imu_ref):
        self.min_t = None
        self.max_t = None
        super().__init__()

        self.traj       = traj
        self.cam_traj   = cam_traj
        self.imu_ref    = imu_ref

    @show_plot
    def plot(self, config, t_end):
        super().plot()

        self.min_t = config.min_t
        self.max_t = min(self.traj.t[-1], config.max_t, t_end)

        self._plot_imu_states(config)
        self._plot_camera_states(config)

    def _plot_core(self, labels, num_cols, offset, filename='',
            cam=None, imu_ref=None, imu_recon=None, axes=None):
        num_rows = math.ceil( len(labels) / num_cols )
        axes = self._init_figure(axes, num_rows, num_cols)

        ai = offset
        for i, label in enumerate(labels):
            if label in ['vx', 'rx', 'dof1', 'dof4', 'rxc']:
                ai += 1

            row, col = self._get_plot_rc(ai, num_rows)
            a = axes[row][col]
            a.set_visible(True)

            val_filt   = self.traj.__dict__[label]
            val_cam    = cam.__dict__[label[:-1]] if cam \
                                else []
            val_imuref = imu_ref.__dict__[label] if \
                            (imu_ref and 'dof' not in label) \
                                else []
            val_recon  = imu_recon.__dict__[label] \
                            if (imu_recon and label in imu_recon.__dict__) \
                                else []

            self._plot_vals(self.traj, val_filt, a)
            self._plot_vals(cam, val_cam, a)
            self._plot_vals(imu_ref, val_imuref, a)
            self._plot_vals(imu_recon, val_recon, a)

            self._ax_postfix(a, label, *val_filt, *val_cam,
                                    *val_imuref, *val_recon)
            ai += 1

        # late setting of line styles
        self._set_line_styles(axes)

        # position legend near first plot
        r0, c0 = self._get_plot_rc(offset, num_rows)
        axes[r0][c0].legend(bbox_to_anchor=(1., 2.))

        self.save(filename)

        return axes

    def _plot_vals(self, val_container, vals, ax):
        if vals != []:
            ax.plot(val_container.t, vals,
                    label=val_container.name)

    def _plot_imu_states(self, config, imu_recon=None,
            axes=None):
        labels = self.traj.labels_imu + self.traj.labels_imu_dofs
        num_cols = 6

        offset = len(labels) % 2
        return self._plot_core(labels, num_cols, offset,
            imu_ref=self.imu_ref, imu_recon=imu_recon,
            filename=config.img_filepath_imu, axes=axes)

    def _plot_camera_states(self, config, axes=None):
        labels = self.traj.labels_camera
        num_cols = 3
        offset = 1

        return self._plot_core(labels, num_cols, offset, cam=self.cam_traj,
            filename=config.img_filepath_cam, axes=axes)

    def _set_plot_line_format(self, line):
        super()._set_plot_line_format(line)

        label = line.get_label()
        if label == 'kf':
            self._apply_lf_dict(line, kf_line_format)
        elif label == 'imu ref':
            self._apply_lf_dict(line, imuref_line_format)
        else:
            self._apply_lf_dict(line, default_line_format)
