import numpy as np
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3
    
def traj_gen(z0, zT, t, t_prev, T):
    t = t - t_prev
    T = T - t_prev

    return z0 + (zT - z0) * (35*(t/T)**4 - 84*(t/T)**5 + 70*(t/T)**6 - 20*(t/T)**7)

def gen_t_partition(t, max_vals, partitions):
    t_part = [0] * (len(partitions)-1)
    p_prev = 0
    for i, p in enumerate(partitions[1:]):
        p_k = int(np.ceil(p * max_vals))
        t_part[i] = np.array(t[p_prev : p_k])
        p_prev = p_k
    return t_part

def gen_values(t, vals, max_vals, partitions):
    vprev = vals[0]
    t_part = gen_t_partition(t, max_vals, partitions)
    t_prev = t_part[0][0]

    traj = [0] * len(t_part)

    for i, vk in enumerate(vals[1:]):
        t_max = t_part[i][-1]
        traj[i] = traj_gen(vprev, vk, t_part[i], t_prev, t_max)
        t_prev = t_max
        vprev = vk

    return np.concatenate(traj).ravel()

def gen_values_rot(t, vals, max_vals, dir, partitions):
    directions = ['x', 'y', 'z']
    assert(dir in directions)

    qx, qy, qz = 0 * t, 0 * t, 0 * t
    qw = 0 * t + 1
    
    angs = gen_values(t, vals, max_vals, partitions)

    for i, ang in enumerate(angs):
        rot_mat = eval(f'SE3.R{dir}({ang}, \'deg\').R')
        qx[i], qy[i], qz[i], qw[i] = R.from_matrix(rot_mat).as_quat()

    return qx, qy, qz, qw

def write_to_file(filepath, t, x, y, z, qx, qy, qz, qw):
    with open(filepath, 'w+') as f:
        for i, ti in enumerate(t):
            data_str = f'{ti:.0f} {x[i]:.3f} {y[i]:.3f} {z[i]:.3f} {qx[i]:.3f} {qy[i]:.3f} {qz[i]:.3f} {qw[i]:.3f}'
            f.write(data_str + '\n')

if __name__ == '__main__':
    filepath = 'trajs/smooth_cam_traj.txt'
    num_frames = 60
    t = np.linspace(0, num_frames-1, num_frames)
    partitions = np.array([0, 0.17, 1])

    # translations
    xvals = 0 * np.array([0, 1, 1])
    yvals = 0 * np.array([0, 1, 1])
    zvals = 0 * np.array([0, 1, 1])

    x = gen_values(t, xvals, num_frames, partitions)
    y = gen_values(t, yvals, num_frames, partitions)
    z = gen_values(t, zvals, num_frames, partitions)

    # rotations
    rotvals = 0 * np.array([0, 1, 1])
    qx, qy, qz, qw = gen_values_rot(t, rotvals, num_frames, 'z', partitions)
    # qx, qy, qz, qw = rotate('x', t, 0, 0*270*3)

    write_to_file(filepath, t, x, y, z, qx, qy, qz, qw)