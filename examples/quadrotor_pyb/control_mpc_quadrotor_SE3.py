import logging
import math
import time

import gym
import numpy as np
import torch
import torch.autograd
import envs
from gym import wrappers, logger as gym_log
from mpc import mpc
from LieFVIN import SE3FVIN, from_pickle
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from controller_utils import *
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu') #
dt = 0.02
mpc_dt = 0.02
requires_grad = True

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')




def get_traj_s(obs, t_curr):
    state = obs['0']["state"]
    quat = state[3:7]
    R = Rotation.from_quat(quat)
    rotmat = R.as_matrix()
    # Save current drone state for plotting
    w_bodyframe = np.matmul(rotmat.T, state[13:16])
    #v_worldframe = state[10:13]
    curr_traj_s = np.zeros(14)
    curr_traj_s[0:3] = state[0:3] # position
    curr_traj_s[3:6] = state[10:13] # linear velocity in world frame
    curr_traj_s[6:9] = np.array([0, 0, 0]) # linear acceleration
    curr_traj_s[9] = state[9] # yaw angles
    curr_traj_s[10:13] = w_bodyframe[:] # angular velocity
    curr_traj_s[-1] = t_curr # time step
    mpc_state = np.concatenate((curr_traj_s[0:3], rotmat.flatten(), curr_traj_s[3:6], w_bodyframe))
    return curr_traj_s, mpc_state

def get_plan_s(traj, INIT_XYZS, t_curr):
    s_plan = traj(t_curr)
    curr_plan_s = np.zeros(11)
    curr_plan_s[0:3] = s_plan.pos + INIT_XYZS.flatten()
    curr_plan_s[3:6] = s_plan.vel
    curr_plan_s[6:9] = s_plan.acc
    curr_plan_s[9] = s_plan.yaw
    curr_plan_s[-1] = t_curr
    Rd = Rotation.from_euler("xyz", [0, 0, s_plan.yaw])
    rotmatd = Rd.as_matrix()
    wd_bodyframe = np.array([0., 0., 0.])
    desired_mpc_state = np.concatenate((curr_plan_s[0:3], rotmatd.flatten(), curr_plan_s[3:6], wd_bodyframe))

    return curr_plan_s, desired_mpc_state

def get_mpc_ref(traj, INIT_XYZS, t_curr, goal_weights):
    all_p = None
    all_goals = None
    for j in range(TIMESTEPS):
        _, desired_mpc_state = get_plan_s(traj, INIT_XYZS, t_curr + j * mpc_dt)
        desired_mpc_state = torch.tensor(desired_mpc_state, device=device, dtype=torch.float64)
        goal_state = torch.squeeze(desired_mpc_state)  # torch.tensor((0., 0.))  # nx
        px = -torch.sqrt(goal_weights) * goal_state
        p = torch.cat((px, torch.zeros(nu, device=device)))
        if all_p is None:
            all_p = p[None, None, :]
            all_goals = goal_state[None, None, :]
        else:
            all_p = torch.cat((all_p, p[None, None, :]), dim=0)
            all_goals = torch.cat((all_goals, goal_state[None, None, :]), dim = 0)
    return all_p, all_goals


if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 10  # T
    N_BATCH = 1
    LQR_ITER = 5


    class QuadrotorDynamicsWrapper(torch.nn.Module):
        def __init__(self):
            super(QuadrotorDynamicsWrapper, self).__init__()
            self.gt = False
            self.model, _ = self.get_model()

        def get_model(self):
            model = SE3FVIN(device=device, time_step=dt).to(device)
            stats = None
            if not self.gt:
                path = './data/run1/quadrotor-se3fvin-vin-5p5-40000.tar'
                model.load_state_dict(torch.load(path, map_location=device))
                path = './data/run1/quadrotor-se3fvin-vin-5p-stats.pkl'
                stats = from_pickle(path)
            return model, stats

        def forward(self, state, action):
            curx = torch.cat((state, action), dim=1)
            if not curx.requires_grad:
                curx.requires_grad = True
            step_num = int(mpc_dt/dt)
            for i in range(step_num):
                if self.gt:
                    nextx = self.model.forward_gt(curx)
                else:
                    nextx = self.model.forward2(curx)
                    curx = nextx
            state, _ = torch.split(nextx, [self.model.posedim + self.model.twistdim, self.model.udim], dim=1)
            return state


    # def angle_normalize(x):
    #     return (((x + math.pi) % (2 * math.pi)) - math.pi)

    ############################################################################################################
    # Initial position and orientation of the drone
    INIT_XYZS = np.array([0.0, 0.0, 0.2]).reshape(1, 3)
    INIT_RPYS = np.array([0, 0, 30 * (np.pi / 180)]).reshape(1, 3)
    # Get desired trajectory (diamond-shaped)
    h_traj = diamond

    t_start = 0  # start of simulation in seconds
    t_step = dt

    # Start to track the desired state for plotting
    s_plan = []
    s_traj = []


    # Start pybullet drone environment
    env = CtrlAviary(drone_model=DroneModel.CF2P,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=Physics.PYB,
                     gui=True,
                     record=False,
                     obstacles=False,
                     user_debug_gui=False,
                     freq=int(1/dt)
                     )

    # Get PyBullet's and drone's ids
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()

    # Draw base frame on the drone
    env._showDroneLocalAxes(0)

    # Start the simulation with no control
    action = {'0': np.array([0, 0, 0, 0])}
    START = time.time()
    t_curr = 0.0
    #obs, reward, done, info = env.step(action)
    obs = env.get_obs()





    # Build conversion matrix betweek force/torch and the motor speeds.
    # We need this because the pybullet drone environment takes motor speeds as input,
    # while in our setting, the control input is force/torch.
    r = env.KM / env.KF
    conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                               [0.0, env.L, 0.0, -env.L],
                               [-env.L, 0.0, env.L, 0.0],
                               [-r, r, -r, r]])
    conversion_mat_inv = np.linalg.inv(conversion_mat)
    ############################################################################################################
    nx = 18
    nu = 4

    u_init = None
    render = True
    retrain_after_iter = 100
    tracking_time = 25
    run_iter = int(tracking_time/mpc_dt)

    wgtR = 0.00001
    goal_weights = torch.tensor((1.2, 1.2, 1.2, wgtR, wgtR, wgtR, wgtR, wgtR, wgtR, wgtR, wgtR, wgtR, 1.2, 1.2, 1.2, 0.0001, 0.0001, 0.0001), \
                                device=device)  #

    ctrl_penalty = 0.000001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(nu, device=device)
    ))  # nx + nu
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu

    # control limits
    ACTION_LOW = torch.tensor([0, -env.MAX_XY_TORQUE, -env.MAX_XY_TORQUE, -env.MAX_Z_TORQUE], device=device)
    ACTION_LOW = ACTION_LOW.repeat(TIMESTEPS, N_BATCH, 1)
    ACTION_HIGH = torch.tensor([env.MAX_THRUST, env.MAX_XY_TORQUE, env.MAX_XY_TORQUE, env.MAX_Z_TORQUE], device=device)
    ACTION_HIGH = ACTION_HIGH.repeat(TIMESTEPS, N_BATCH, 1)

    # run MPC
    total_reward = 0
    state_traj = []
    for i in range(run_iter):
        print(i)
        p, all_goals = get_mpc_ref(h_traj, INIT_XYZS, i*mpc_dt, goal_weights)
        cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

        curr_plan_s, _ = get_plan_s(h_traj, INIT_XYZS, t_curr=i*mpc_dt)
        s_plan.append(curr_plan_s)
        curr_traj_s, mpc_state = get_traj_s(obs, t_curr=i*mpc_dt)
        s_traj.append(curr_traj_s)

        # print("mpc_state:", p.shape, "\n", p[:5,0,0:12])
        state = torch.tensor(mpc_state, device=device, requires_grad=requires_grad).view(1, -1)
        command_start = time.perf_counter()
        # recreate controller using updated u_init (kind of wasteful right?)
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-3,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)

        # compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, QuadrotorDynamicsWrapper())
        first_action = nominal_actions[0]  # take first planned action
        next_last_action = torch.clone(nominal_actions[-1])
        u_init = torch.cat((nominal_actions[1:], next_last_action[None,:,:]), dim=0)

        elapsed = time.perf_counter() - command_start
        thrust_torques = first_action.detach().cpu().numpy()

        rpm_squared = np.matmul(conversion_mat_inv, thrust_torques.reshape(4,1))
        rpm_squared = rpm_squared[:,0]
        rpm_squared[0] = max(0, rpm_squared[0])
        rpm_squared[1] = max(0, rpm_squared[1])
        rpm_squared[2] = max(0, rpm_squared[2])
        rpm_squared[3] = max(0, rpm_squared[3])
        rpm_from_thrusttorques = np.sqrt(rpm_squared / env.KF)
        action['0'] = rpm_from_thrusttorques
        #############################
        obs, r, _, _ = env.step(action)

        total_reward += r
        print("Desired pos: ", curr_plan_s[0:3], curr_plan_s[3:6])
        logger.debug("action taken: [%.4f %.6f %.6f %.6f]  cost received: %.4f time taken: %.5fs", thrust_torques[0,0], thrust_torques[0,1], thrust_torques[0,2], thrust_torques[0,3], -r, elapsed)

        if render:
            env.render()

    s_traj = np.array(s_traj)
    s_plan = np.array(s_plan)
    plot_states1D(s_traj, s_plan)
    quadplot_update(s_traj, s_plan)

    logger.info("Total reward %f", total_reward)
    # Plot trajectory
    env.close()

    np.savez("./ToPlot/mpc_data.npz", s_traj=s_traj, s_plan=s_plan, goal_weights=goal_weights.detach().cpu().numpy(), ctrl_penalty=ctrl_penalty)