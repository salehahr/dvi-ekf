import numpy as np
from .context import VisualTraj

class Interpolator(object):
    def __init__(self, interframe_vals: int,
            uninterp_traj: VisualTraj):
        name_interp     = uninterp_traj.name + ' interpl'

        self.interframe_vals    = interframe_vals
        self.uninterp_traj      = uninterp_traj
        self.interp_traj        = VisualTraj(name_interp)

        self.t_old = self.uninterp_traj.t
        self.t_new = self._gen_t_new()

    def _gen_t_new(self) -> np.array:
        """ Generates time series for the interpolated data. """
        tmin   = self.t_old[0]
        tmax   = self.t_old[-1]

        num_old_datapoints = len(self.t_old)
        num_new_datapoints = (num_old_datapoints - 1) * self.interframe_vals + 1

        return np.linspace(tmin, tmax, num=num_new_datapoints)

    def interpolate(self):
        """ Interpolates data and stores it in self.interp_traj.
            Also generates quats_array in self.interp_traj.
        """
        for label in self.uninterp_traj.labels:
            self._interpolate_label(label)
        self.flag_done = True
        self.interp_traj._gen_quats_farray()

    def _interpolate_label(self, label: str):
        """ Interpolate data corresponding to a single label. """
        if label == 't':
            self.interp_traj.t = self.t_new
            return

        vals_old = self.uninterp_traj.__dict__[label]
        vals_new = np.interp(self.t_new, self.t_old, vals_old)

        self.interp_traj.__dict__[label] = vals_new

    @property
    def flag_done(self):
        return self.interp_traj.is_interpolated

    @flag_done.setter
    def flag_done(self, val: bool):
        if val is True:
            self.interp_traj.interframe_vals = self.interframe_vals
            self.interp_traj.is_interpolated = True

    @property
    def interpolated(self):
        if not self.flag_done:
            self.interpolate()
        return self.interp_traj