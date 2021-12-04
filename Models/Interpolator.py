import numpy as np

from .context import VisualTraj


class Interpolator(object):
    def __init__(self, interframe_vals: int, uninterp_camera: "Camera"):
        name_interp = uninterp_camera.traj.name + " interpl"

        self.interframe_vals = interframe_vals
        self.uninterp_camera = uninterp_camera
        self.uninterp_traj = uninterp_camera.traj
        self.interp_traj = VisualTraj(name_interp)

        self.t_old = self.uninterp_traj.t
        self.t_new = self._gen_t_new()

    def _gen_t_new(self) -> np.array:
        """Generates time series for the interpolated data."""
        tmin = self.t_old[0]
        tmax = self.t_old[-1]

        num_old_datapoints = len(self.t_old)
        num_new_datapoints = (num_old_datapoints - 1) * self.interframe_vals + 1

        return np.linspace(tmin, tmax, num=num_new_datapoints)

    def interpolate(self):
        """Interpolates data and stores it in self.interp_traj.
        Also generates quats_array in self.interp_traj.
        """
        # trajectory labels
        for label in self.uninterp_traj.labels:
            self._interpolate_traj_label(label)

        # interpolate values not in Trajectory
        cam_labels = ["v", "acc", "om", "alp"]
        for label in cam_labels:
            self._interpolate_cam_label(label)

        # interpolate notch values
        if self.uninterp_camera.with_notch:
            notch_labels = ["notch", "notch_d", "notch_dd"]
            for label in notch_labels:
                self._interpolate_notch_label(label)

        self.flag_done = True
        self.interp_traj._gen_quats_array()

    def _interpolate_traj_label(self, label: str):
        """Interpolate data corresponding to a single label."""
        if label == "t":
            self.interp_traj.t = self.t_new
            return

        vals_old = self.uninterp_traj.__dict__[label]
        vals_new = np.interp(self.t_new, self.t_old, vals_old)

        self.interp_traj.__dict__[label] = vals_new

    def _interpolate_cam_label(self, label: str):
        """Interpolate data corresponding to a single label."""
        component_values = {"x": [], "y": [], "z": []}

        for i, k in enumerate(component_values):
            vals_old = self.uninterp_camera.__dict__[label][i, :]
            component_values[k] = np.interp(self.t_new, self.t_old, vals_old)

        vals_new = np.asarray(
            (component_values["x"], component_values["y"], component_values["z"])
        )

        self.interp_traj.__dict__[label] = vals_new

    def _interpolate_notch_label(self, label: str):
        """Interpolate data corresponding to a single label."""
        assert self.uninterp_camera.notch is not None
        vals_old = self.uninterp_camera.__dict__[label]
        assert vals_old is not None
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
