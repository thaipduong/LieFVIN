import time
import argparse
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from LieFVIN import to_pickle, from_pickle

# Pybullet drone environment
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
#from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from envs import Pybullet_Quadrotor

parser = argparse.ArgumentParser(description='Debugging script for PyBullet applyExternalForce() and applyExternalTorque() PyBullet')
parser.add_argument('--duration_sec',   default=3,     type=float,       help='Duration of the simulation in seconds (default: 30)', metavar='')
parser.add_argument('--num_resets',     default=1,      type=int,       help='Number of times the simulation is reset to its initial conditions (default: 2)', metavar='')
parser.add_argument('--simulation_freq_hz', default=50, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
#parser.add_argument('--control_freq_hz', default=48, type=int, help='Control frequency in Hz (default: 48)', metavar='')
ARGS = parser.parse_args()



def sample_gym(sim_freq = 240, ctrl_freq = 48):
    AGGR_PHY_STEPS = int(sim_freq / ctrl_freq)
    AGGR_SIM_STEPS = int(ARGS.simulation_freq_hz/sim_freq)
    # A set of goal positions
    x = np.arange(-1, 2, 1)
    y = np.arange(-1, 2, 1)
    z = 0.75*np.arange(0.5, 1.5, 0.5)
    xx, yy, zz = np.meshgrid(x, y, z)
    ITER_NUM = len(x) * len(y) * len(z)
    SET_POINTS = np.zeros([ITER_NUM, 3])
    SET_POINTS[:, 0] = np.reshape(xx, (ITER_NUM,))
    SET_POINTS[:, 1] = np.reshape(yy, (ITER_NUM,))
    SET_POINTS[:, 2] = np.reshape(zz, (ITER_NUM,))
    state_control_dim = 18 + 4 # 18 for states (3 for x, 9 for R, 3 for v, 3 for w), 4 for control
    data_set = np.zeros((ITER_NUM*2, AGGR_PHY_STEPS, int(ARGS.duration_sec*ctrl_freq), state_control_dim))
    # Random initialized position
    init_pos = 0.0*np.array([[-1.0, -1.0, 1.0, 1.0],
                             [-1.0, 1.0, -1.0, 1.0]]) + np.random.normal(0.0, 0.01, size=(2, 4))
    for iter in range(ITER_NUM):
        # Pick Target position and orientation
        INIT_XYZS = np.array([init_pos[0,iter%4],init_pos[1,iter%4], 0.0]).reshape(1, 3)
        TARGET_POS = INIT_XYZS + SET_POINTS[iter, :].reshape(1, 3)
        INIT_RPYS = np.array([0, 0, (iter%3)*30*(np.pi/180)]).reshape(1, 3)
        TARGET_RPYS = np.array([0, 0, (iter % 3) * 15 * (np.pi / 180)]).reshape(1, 3)

        # Initialize the simulation
        env = Pybullet_Quadrotor(drone_model=DroneModel.CF2P,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         gui=True,
                         record = False,
                         obstacles=False,
                         user_debug_gui=False,
                         freq=ARGS.simulation_freq_hz
                         )

        # Get PyBullet's and drone's ids
        PYB_CLIENT = env.getPyBulletClient()
        DRONE_IDS = env.getDroneIds()
        env._showDroneLocalAxes(0)

        # PID control for set point regulation
        ctrl = DSLPIDControl(drone_model=DroneModel.CF2P)
        # Setting control gains
        ctrl.P_COEFF_FOR = 0.1 * np.array([.4, .4, 1.25])
        ctrl.I_COEFF_FOR = 0.1 * np.array([.05, .05, .05])
        ctrl.D_COEFF_FOR = 0.1 * np.array([.2, .2, .5])
        ctrl.P_COEFF_TOR = 0.1 * np.array([70000., 70000., 60000.])
        ctrl.I_COEFF_TOR = 0.1 * np.array([.0, .0, 500.])
        ctrl.D_COEFF_TOR = 0.1 * np.array([20000., 20000., 12000.])

        # Conversion matrix between motor speeds and thrust and torques.
        r = env.KM / env.KF
        conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                                   [0.0, env.L, 0.0, -env.L],
                                   [-env.L, 0.0, env.L, 0.0],
                                   [-r, r, -r, r]])
        conversion_mat_inv = np.linalg.inv(conversion_mat)

        # Run the simulation
        START = time.time()
        action = {'0': np.array([0, 0, 0, 0])}
        obs, reward, done, info = env.step(action)


        for i in range(int(ARGS.duration_sec*ctrl_freq)):
            # The action from the PID controller is motor speed for each motor.
            action['0'], _, _ = ctrl.computeControlFromState(control_timestep=1/ctrl_freq,
                                                             state=obs['0']["state"],
                                                             target_pos=TARGET_POS[0,:],
                                                             target_rpy=TARGET_RPYS[0,:])
            # print("Target pos: ", TARGET_POS[0,0:2], INIT_XYZS[0, 2])
            # print("Target rpy: ",TARGET_RPYS[0,:])
            # We convert motor speed to thrust and force for data collection.


            # Apply the control input for a few time steps
            for s in range(AGGR_PHY_STEPS):
                for internal_step in range(AGGR_SIM_STEPS):
                    obs, reward, done, info = env.step(action)
                clipped_action = env.last_clipped_action
                rpm = clipped_action[0]
                forces = np.array(rpm ** 2) * env.KF
                thrust_torques = np.matmul(conversion_mat, forces)
                # The state constains: position (3), quaternion (4), rpy (3), linear vel (3), angular vel (3), motor speed rpm (4)
                state = obs['0']["state"]
                quat = state[3:7]
                R = Rotation.from_quat(quat)
                rotmat = R.as_matrix()
                v_worldframe = state[10:13]
                v_bodyframe = np.matmul(rotmat.T,  v_worldframe)
                w_bodyframe = np.matmul(rotmat.T, state[13:16])
                collected_state = np.concatenate((state[0:3], rotmat.flatten(), v_worldframe, w_bodyframe, thrust_torques))
                data_set[iter, s, i, :] = collected_state
                env.render()
            sync(i, START, env.TIMESTEP*AGGR_PHY_STEPS*AGGR_SIM_STEPS)
        # Close the environment
        env.close()
    #################################################################################################################
    ##### SWAP THE GOAL AND INIT TO FLY THE DRONE DOWNWARD##########################
    for iter in range(ITER_NUM, 2*ITER_NUM):
        # Pick Target position and orientation
        INIT_XYZS = np.array([init_pos[0, iter % 4], init_pos[1, iter % 4], 0.75]).reshape(1, 3)
        TARGET_POS = INIT_XYZS + SET_POINTS[iter%ITER_NUM, :].reshape(1, 3) + np.array([0., 0., 1.0]).reshape(1,3)
        INIT_RPYS = np.array([0, 0, (iter % 3) * 30 * (np.pi / 180)]).reshape(1, 3)
        TARGET_RPYS = np.array([0, 0, (iter % 3) * 15 * (np.pi / 180)]).reshape(1, 3)

        # SWAP
        TEMP = INIT_XYZS
        INIT_XYZS = TARGET_POS
        TARGET_POS = TEMP

        TEMP = INIT_RPYS
        INIT_RPYS = TARGET_RPYS
        TARGET_RPYS = TEMP


        # Initialize the simulation
        env = Pybullet_Quadrotor(drone_model=DroneModel.CF2P,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         gui=True,
                         record=False,
                         obstacles=False,
                         user_debug_gui=False,
                         freq=ARGS.simulation_freq_hz
                         )

        # Get PyBullet's and drone's ids
        PYB_CLIENT = env.getPyBulletClient()
        DRONE_IDS = env.getDroneIds()
        env._showDroneLocalAxes(0)

        # PID control for set point regulation
        ctrl = DSLPIDControl(drone_model=DroneModel.CF2P)
        # Setting control gains
        ctrl.P_COEFF_FOR = 0.1 * np.array([.4, .4, 1.25])
        ctrl.I_COEFF_FOR = 0.1 * np.array([.05, .05, .05])
        ctrl.D_COEFF_FOR = 0.1 * np.array([.2, .2, .5])
        ctrl.P_COEFF_TOR = 0.1 * np.array([70000., 70000., 60000.])
        ctrl.I_COEFF_TOR = 0.1 * np.array([.0, .0, 500.])
        ctrl.D_COEFF_TOR = 0.1 * np.array([20000., 20000., 12000.])

        # Conversion matrix between motor speeds and thrust and torques.
        r = env.KM / env.KF
        conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                                   [0.0, env.L, 0.0, -env.L],
                                   [-env.L, 0.0, env.L, 0.0],
                                   [-r, r, -r, r]])
        conversion_mat_inv = np.linalg.inv(conversion_mat)

        # Run the simulation
        START = time.time()
        action = {'0': np.array([0, 0, 0, 0])}
        obs, reward, done, info = env.step(action)

        for i in range(int(ARGS.duration_sec * ctrl_freq)):
            # The action from the PID controller is motor speed for each motor.
            action['0'], _, _ = ctrl.computeControlFromState(control_timestep=1 / ctrl_freq,
                                                             state=obs['0']["state"],
                                                             target_pos=TARGET_POS[0, :],
                                                             target_rpy=TARGET_RPYS[0, :])
            # print("Target pos: ", TARGET_POS[0,0:2], INIT_XYZS[0, 2])
            # print("Target rpy: ",TARGET_RPYS[0,:])
            # We convert motor speed to thrust and force for data collection.

            # Apply the control input for a few time steps
            for s in range(AGGR_PHY_STEPS):
                for internal_step in range(AGGR_SIM_STEPS):
                    obs, reward, done, info = env.step(action)
                clipped_action = env.last_clipped_action
                rpm = clipped_action[0]
                forces = np.array(rpm ** 2) * env.KF
                thrust_torques = np.matmul(conversion_mat, forces)
                # The state constains: position (3), quaternion (4), rpy (3), linear vel (3), angular vel (3), motor speed rpm (4)
                state = obs['0']["state"]
                quat = state[3:7]
                R = Rotation.from_quat(quat)
                rotmat = R.as_matrix()
                v_worldframe = state[10:13]
                v_bodyframe = np.matmul(rotmat.T, v_worldframe)
                w_bodyframe = np.matmul(rotmat.T, state[13:16])
                collected_state = np.concatenate(
                    (state[0:3], rotmat.flatten(), v_worldframe, w_bodyframe, thrust_torques))
                data_set[iter, s, i, :] = collected_state
                env.render()
            sync(i, START, env.TIMESTEP * AGGR_PHY_STEPS * AGGR_SIM_STEPS)
        # Close the environment
        env.close()
    tspan = np.arange(AGGR_PHY_STEPS)/sim_freq
    return data_set, tspan

def get_dataset(test_split=0.5, save_dir=None, **kwargs):
    data = {}

    assert save_dir is not None
    path = '{}/pybullet-drone-dataset160.pkl'.format(save_dir)
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        data_set, tspan = sample_gym(**kwargs)
        # Make a train/test split
        samples = data_set.shape[2]
        split_ix = int(samples * test_split)
        rng = np.random.default_rng()
        idx = np.arange(data_set.shape[2])
        rng.shuffle(idx)
        train_idx = np.sort(idx[:split_ix])
        test_idx = np.sort(idx[split_ix:])

        split_data = {}
        split_data['x'], split_data['test_x'] = data_set[:,:,train_idx,:], data_set[:,:,test_idx,:]
        data = split_data
        data['t'] = tspan
        to_pickle(data, path)
    return data


def arrange_data(x, t, num_points=2):
    '''Arrange data to feed into neural ODE in small chunks'''
    assert num_points>=2 and num_points<=len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points-1:
            x_stack.append(x[:, i:-num_points+i+1,:,:])
        else:
            x_stack.append(x[:, i:,:,:])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack,
                (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]
    return x_stack, t_eval