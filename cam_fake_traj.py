import numpy as np
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3
from Filter import Quaternion 

def get_index_at(T, t):
    """ Get index for which timestamp matches the argument T. """
    return max([i for i, ti in enumerate(t) if ti <= T])

def make_section(t, start_t, end_t, start_val, end_val):
    i_start = get_index_at(start_t, t)
    i_end = get_index_at(end_t, t)
    i_max = t.shape[0] - 1

    x_prev = np.zeros((i_start,)) if start_t > 0 else np.array([])
    x_during = np.linspace(start_val, end_val, i_end - i_start + 1)
    x_after = np.zeros((i_max - i_end,)) + end_val
    
    assert(x_prev.shape[0] + x_during.shape[0] + x_after.shape[0] == i_max + 1)
    
    arr = np.hstack([x_prev, x_during, x_after])
    assert(arr.shape[0] == t.shape[0])
    
    return arr

def translate(dir, t, start_t, end_t, start_val, end_val):
    directions = ['x', 'y', 'z']
    assert(dir in directions)

    x, y, z = 0*t, 0*t, 0*t

    val = make_section(t, start_t, end_t, start_val, end_val)
    assert(val.shape[0] == t.shape[0])

    return val

def rotate(dir, t, start_val, end_val):
    directions = ['x', 'y', 'z']
    assert(dir in directions)

    qx, qy, qz = 0 * t, 0 * t, 0 * t
    qw = 0 * t + 1

    ang_ramp = np.linspace(start_val, end_val, t.shape[0])
    
    for i, ang in enumerate(ang_ramp):
        rot_mat = eval(f'SE3.R{dir}({ang}, \'deg\').R')
        qx[i], qy[i], qz[i], qw[i] = R.from_matrix(rot_mat).as_quat()
        # qx[i], qy[i], qz[i], qw[i] = Quaternion(val=rot_mat, do_normalise=True).xyzw
    
    return qx, qy, qz, qw

def write_to_file(filepath, t, x, y, z, qx, qy, qz, qw):
    with open(filepath, 'w+') as f:
        for i, ti in enumerate(t):
            data_str = f'{ti:.0f} {x[i]:.3f} {y[i]:.3f} {z[i]:.3f} {qx[i]:.3f} {qy[i]:.3f} {qz[i]:.3f} {qw[i]:.3f}'
            f.write(data_str + '\n')

if __name__ == '__main__':
    filepath = 'trajs/simple_cam_traj.txt'
    num_frames = 60
    t = np.linspace(0, num_frames-1, num_frames)

    # translations
    x = translate('x', t, 0, 10, 0, 0)
    y = translate('y', t, 0, 10, 0, 0)
    z = translate('z', t, 0, 10, 0, 0)

    # rotations
    qx, qy, qz, qw = rotate('x', t, 0, 270*3)

    write_to_file(filepath, t, x, y, z, qx, qy, qz, qw)