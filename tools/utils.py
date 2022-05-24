import numpy as np


def generate_imudof_ic(gt_dofs: np.ndarray) -> np.ndarray:
    """
    Generates synthetic initial conditions of the IMU dofs
    by perturbing the real DOFs

    e.g.
    DOFs (real) : [ 0.    0.  0.  0.  0. 20.]
    DOFs (IC)   : [ 0.05  0.  0.  3.  3. 23.]
    """
    # dofs0_rot_real = gt_dofs[:3]
    # dofs0_tr_real = gt_dofs[3:]
    #
    # # perturbations
    # delta_ang_rad = np.deg2rad(3)
    # delta_trans_cm = 3
    #
    # # # random perturbations
    # # delta_dof_rot = np.random.normal(loc=0, scale=delta_ang_rad, size=(3,))
    # # delta_dof_tr = np.random.normal(loc=0, scale=delta_trans_cm, size=(3,))
    #
    # # const. perturbations
    # delta_dof_rot = [delta_ang_rad, *dofs0_rot_real[1:]]
    # delta_dof_tr = [x + delta_trans_cm for x in dofs0_tr_real]
    #
    # return [*delta_dof_rot, *delta_dof_tr]

    # TODO: change this later; simulations currently use perfect initial conditions
    return gt_dofs
