from configuration import Config
from Filter import FilterTraj, ImuRefTraj

filepath_config = './configs.txt'

config          = Config(traj_name='mandala0_mono', pu=True)
config.read(filepath_config)
kf, camera, _   = config.init_filter_objects()

kf.traj         = FilterTraj('kf', config.traj_kf_filepath)
kf.imu.ref      = ImuRefTraj('imu ref', _, config.traj_imuref_filepath)

t = kf.traj.t[-1]
kf.plot(config, t, camera.traj, compact=True)