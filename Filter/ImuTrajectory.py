import os

import numpy as np
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R

from Filter.Measurement import ImuMeasurement
from Filter.Quaternion import Quaternion
from Filter.Trajectory import Trajectory
from Filter.VisualTrajectory import VisualTraj
from Models.Interpolator import Interpolator


class ImuTraj(Trajectory):
    """IMU trajectory containing the acceleration and
    angular velocity measurements."""

    labels = ["t", "ax", "ay", "az", "gx", "gy", "gz"]

    def __init__(
        self,
        name="imu",
        filepath=None,
        vis_data=None,
        cap=None,
        interframe_vals=0,
        covariance=[0.0] * 6,
        unnoised=False,
    ):

        # base properties
        super().__init__(name, filepath, cap)
        self.vis_data = vis_data

        # extra data
        self.noisy = None
        self.reconstructed = None

        # flags
        self._flag_gen_unnoisy_imu = False

        # data generation
        if vis_data:
            self.reset()
            self._gen_unnoisy_imu()

            if not unnoised:
                self._gen_noisy_imu(covariance)

        self.next_frame_index = 0
        self.queue_first_ts = 0

    def _gen_unnoisy_imu(self):
        """Generates IMU data from visual trajectory, without noise.
        Involves transformation from SLAM world coordinates
        to IMU coordinates."""

        interpolated = Interpolator(self.interframe_vals, self.vis_data).interpolated
        dt = interpolated.t[1] - interpolated.t[0]

        # get rotations for coordinate transformation
        R_BW_arr = [q.rot.T for q in interpolated.quats]

        # velocities
        v_W = np.asarray(
            (
                np.gradient(interpolated.x, dt),
                np.gradient(interpolated.y, dt),
                np.gradient(interpolated.z, dt),
            )
        ).T
        self.vis_data.vx = v_W[:, 0]  # VisTraj is in world coordinates
        self.vis_data.vy = v_W[:, 1]
        self.vis_data.vz = v_W[:, 2]

        v_B = np.asarray(
            [R_BW @ v_W[i, :] for i, R_BW in enumerate(R_BW_arr)]
        )  ## possibly wrong transformation?
        self.vx, self.vy, self.vz = v_B[:, 0], v_B[:, 1], v_B[:, 2]

        # accelerations
        a_W = np.array(
            (
                np.gradient(v_W[:, 0], dt),
                np.gradient(v_W[:, 1], dt),
                np.gradient(v_W[:, 2], dt),
            )
        ).T

        a_B = np.asarray([R_BW @ a_W[i, :] for i, R_BW in enumerate(R_BW_arr)])
        self.ax, self.ay, self.az = a_B[:, 0], a_B[:, 1], a_B[:, 2]

        # angular velocity
        euler = np.asarray([q.euler_zyx for q in interpolated.quats])
        rz, ry, rx = euler[:, 0], euler[:, 1], euler[:, 2]

        om_W = np.asarray(
            (np.gradient(rx, dt), np.gradient(ry, dt), np.gradient(rz, dt))
        )
        om_B = np.asarray([R_BW @ om_W[:, i] for i, R_BW in enumerate(R_BW_arr)]).T
        self.gx, self.gy, self.gz = om_B[0, :], om_B[1, :], om_B[2, :]

        self.t = interpolated.t

        if self.filepath:
            self._write_to_file()

        self._flag_gen_unnoisy_imu = True

    def _gen_noisy_imu(self, covariance):
        """Generates IMU data from visual trajectory, with noise."""

        assert self._flag_gen_unnoisy_imu == True

        if self.filepath:
            filename, ext = os.path.splitext(self.filepath)
            filename_noisy = filename + "_noisy" + ext
        else:
            filename_noisy = None

        noisy = ImuTraj(
            name="noisy imu",
            filepath=filename_noisy,
            interframe_vals=self.interframe_vals,
            covariance=covariance,
        )
        noisy.reset()

        for i, label in enumerate(self.labels):
            if label == "t":
                noisy.t = self.t
                continue

            noisy.__dict__[label] = self.__dict__[label] + np.random.normal(
                loc=0.0, scale=covariance[i - 1], size=len(self.t)
            )

        self.noisy = noisy

        if self.filepath:
            self._write_to_file(filename_noisy)

    def _write_to_file(self, filename=None):
        """Writes IMU trajectory to file."""

        if filename == None:
            filename = self.filepath

        with open(filename, "w+") as f:
            for i, t in enumerate(self.t):
                a_str = f"{self.ax[i]:.9f} {self.ay[i]:.9f} {self.az[i]:.9f} "
                g_str = f"{self.gx[i]:.9f} {self.gy[i]:.9f} {self.gz[i]:.9f}"
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + "\n")

    def at_index(self, index):
        """Returns single IMU measurement at the given index."""

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
        """Get IMU queue after old_t and up to current_cam_t."""

        start_index = self.next_frame_index

        prev_index = self._get_index_at(old_t)  # end of old imu queue
        next_index = self._get_index_at(current_cam_t)
        assert prev_index <= next_index

        # update start index, if outdated (less than prev_index)
        if (start_index <= prev_index) and not (start_index == next_index):
            start_index = prev_index + 1

        t = self.t[start_index : next_index + 1]

        ax = self.ax[start_index : next_index + 1]
        ay = self.ay[start_index : next_index + 1]
        az = self.az[start_index : next_index + 1]
        acc = np.vstack((ax, ay, az))

        gx = self.gx[start_index : next_index + 1]
        gy = self.gy[start_index : next_index + 1]
        gz = self.gz[start_index : next_index + 1]
        om = np.vstack((gx, gy, gz))

        self.next_frame_index = next_index + 1

        return ImuMeasurement(t, acc, om)

    def reconstruct(self, R_WB, W_p_BW_0, WW_v_BW_0):
        """For validation.
        Generates trajectory from IMU data.
        The IMU trajectory is obtained via numerical integration
        using the available initial conditions."""

        reconstructed = VisualTraj("recon")

        t = self.t
        dt = t[1] - t[0]
        reconstructed.t = t

        # initial conditions in world coordinates
        x0, y0, z0 = W_p_BW_0
        vx0, vy0, vz0 = WW_v_BW_0
        rz0, ry0, rx0 = Quaternion(val=R_WB[0], do_normalise=True).euler_zyx

        # velocity in world coordinates
        assert len(R_WB) == len(self.ax)
        W_acc = self._to_world_coords(R_WB, np.asarray((self.ax, self.ay, self.az)).T)

        W_vx = cumtrapz(W_acc[:, 0], t, initial=0) + vx0
        W_vy = cumtrapz(W_acc[:, 1], t, initial=0) + vy0
        W_vz = cumtrapz(W_acc[:, 2], t, initial=0) + vz0

        # position in world coordinates
        reconstructed.x = cumtrapz(W_vx, t, initial=0) + x0
        reconstructed.y = cumtrapz(W_vy, t, initial=0) + y0
        reconstructed.z = cumtrapz(W_vz, t, initial=0) + z0

        # orientation in world
        W_om_B = self._to_world_coords(R_WB, np.asarray((self.gx, self.gy, self.gz)).T)

        rx = cumtrapz(W_om_B[:, 0], t, initial=0) + rx0
        ry = cumtrapz(W_om_B[:, 1], t, initial=0) + ry0
        rz = cumtrapz(W_om_B[:, 2], t, initial=0) + rz0

        euler_ang = np.asarray([rz, ry, rx]).T
        rots = [R.from_euler("zyx", e).as_matrix() for e in euler_ang]
        quats = [Quaternion(val=R_i, do_normalise=True) for R_i in rots]

        reconstructed.qx = [q.x for q in quats]
        reconstructed.qy = [q.y for q in quats]
        reconstructed.qz = [q.z for q in quats]
        reconstructed.qw = [q.w for q in quats]
        reconstructed.gen_angle_arrays()

        self.reconstructed = reconstructed

    def reconstruct_vis_traj(self):
        """Generates trajectory from IMU data using
        the available initial conditions."""

        reconstructed = VisualTraj("recon")

        t = self.t
        dt = t[1] - t[0]
        reconstructed.t = t

        # get rotations for coordinate transformation
        R_WB_arr = [q.rot for q in self.vis_data.interpolated.quats]

        # initial conditions in world coordinates
        R_WB0 = R_WB_arr[0]
        x0, y0, z0 = self.vis_data.x[0], self.vis_data.y[0], self.vis_data.z[0]
        v0 = self._to_world_coords(
            R_WB0, np.asarray((self.vx[0], self.vy[0], self.vz[0]))
        )
        q0 = self.vis_data.quats[0]

        # integrating for pos
        a_W = self._to_world_coords(R_WB_arr, np.asarray((self.ax, self.ay, self.az)).T)
        vx_W = cumtrapz(a_W[:, 0], t, initial=0) + v0[0]
        vy_W = cumtrapz(a_W[:, 1], t, initial=0) + v0[1]
        vz_W = cumtrapz(a_W[:, 2], t, initial=0) + v0[2]

        reconstructed.x = cumtrapz(vx_W, t, initial=0) + x0
        reconstructed.y = cumtrapz(vy_W, t, initial=0) + y0
        reconstructed.z = cumtrapz(vz_W, t, initial=0) + z0

        # integrating for orientation
        rz0, ry0, rx0 = q0.euler_zyx

        om_B = np.array((self.gx, self.gy, self.gz))
        om_W = np.asarray([R_WB @ om_B[:, i] for i, R_WB in enumerate(R_WB_arr)]).T

        rx = cumtrapz(om_W[0, :], t, initial=0) + rx0
        ry = cumtrapz(om_W[1, :], t, initial=0) + ry0
        rz = cumtrapz(om_W[2, :], t, initial=0) + rz0

        euler_ang = np.asarray([rz, ry, rx]).T
        quats = np.asarray([R.from_euler("zyx", e).as_quat() for e in euler_ang])
        reconstructed.qx = quats[:, 0]
        reconstructed.qy = quats[:, 1]
        reconstructed.qz = quats[:, 2]
        reconstructed.qw = quats[:, 3]

        self.reconstructed = reconstructed

        return reconstructed

    def _set_plot_line_style(self, line):
        """Defines line styles for IMU plot."""

        line.set_linestyle("")
        line.set_marker("o")
        line.set_markersize(0.4)

        label = line.get_label()
        if "noisy" in label:
            line.set_linestyle("-")
            line.set_color("darkgrey")
            line.set_linewidth(1)
        elif "gt" in label:
            line.set_color("black")
        elif "mono" in label:
            line.set_color("saddlebrown")

    def _to_world_coords(self, R_WB, vec):
        if isinstance(R_WB, list):
            assert vec.shape[1] == 3
            return np.asarray([Rot @ vec[i, :] for i, Rot in enumerate(R_WB)])
        else:
            return R_WB @ vec

    def _to_imu_coords(self, R_BW, vec):
        self._to_world_coords(R_BW, vec)
