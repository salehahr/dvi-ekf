import matplotlib.pyplot as plt


def plot_savefig(fig, figname):
    print(f'Saving file "{figname}". ')
    fig.savefig(figname)


def plot_trajectories(config, t_end, kf_traj, camera, imu, do_reconstruct=False):

    max_t = min(kf_traj.t[-1], config.max_t, t_end)

    imu_recon = get_imu_recon_traj(imu) if do_reconstruct else None

    imu_axes = kf_traj.plot_imu(
        config.img_filepath_imu,
        min_t=config.min_t,
        max_t=max_t,
        imu_ref=imu.ref,
        imu_recon=imu_recon,
    )
    cam_axes = kf_traj.plot_camera(
        config.img_filepath_cam, cam=camera.traj, min_t=config.min_t, max_t=max_t
    )

    plt.show()


def get_imu_recon_traj(imu):
    imu.generate_traj("trajs/imu_recon.txt", True)
    imu.reconstruct()
    imu.traj.reconstructed.name = "imu (B) recon"
    return imu.traj.reconstructed


def plot_velocities(kf_traj, do_plot_vel):
    if do_plot_vel:
        axes = stereoGT_traj.plot_velocities()
        axes = kf_traj.plot_velocities(axes, min_t=min_t, max_t=max_t)


def plot_noise_sensitivity(kf_traj, Q, Rp, Rq):
    """plots sensitivity to measurement noise R and process noise Q"""
    axes = stereoGT_traj.plot_sens_noise(Rp, Rq, Q)
    axes = kf_traj.plot_sens_noise(Rp, Rq, Q, axes, min_t=min_t, max_t=max_t)

    figname = f"./img/Rp{Rp}_Rq{Rq}_Q{Q}.png"
    plot_savefig(axes[0].figure, figname)
