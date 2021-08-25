from configuration import Config
from Filter import FilterTraj, ImuRefTraj

filepath     = './trajs/kf_best.txt'
filepath_imu = './trajs/imu_ref.txt'

config          = Config(traj_name='mandala0_mono', pu=True)
kf, camera, _   = config.init_filter_objects()

kf.traj         = FilterTraj('kf', filepath)
kf.imu.ref      = ImuRefTraj('imu ref', _, filepath_imu)

t = kf.traj.t[-1]
kf.plot(config, t, camera.traj)