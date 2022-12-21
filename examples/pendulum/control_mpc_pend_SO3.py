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
from LieFVIN import SO3FVIN, from_pickle
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
gpu=0
device = 'cpu'#torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu') #
dt = 0.02
mpc_dt = 0.02
requires_grad = True

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')



def plot_traj(traj, t_eval):
    ''' Plotting trajectory'''
    # Figure and font size
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    linewidth = 4

    traj = np.array(traj)

    fig = plt.figure(figsize= figsize)
    plt.plot(t_eval, traj[:, 0], label=r'$\varphi$', linewidth=linewidth)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlabel('$t(s)$', fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./png/control_theta.pdf', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize= figsize)
    plt.plot(t_eval, traj[:, 1], label=r'$\dot{\varphi}$', linewidth=linewidth)
    plt.xlabel('$t(s)$', fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./png/control_thetadot.pdf', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, traj[:, -1], label=r'$u$', linewidth=linewidth)
    plt.xlabel('$t(s)$', fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./png/control_input.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 20  # T
    N_BATCH = 1
    LQR_ITER = 5
    ACTION_LOW = -20.0
    ACTION_HIGH = 20.0


    class PendulumDynamicsWrapper(torch.nn.Module):
        def __init__(self):
            super(PendulumDynamicsWrapper, self).__init__()
            self.gt = False
            self.model, _ = self.get_model()


        def get_model(self):
            model = SO3FVIN(device=device, u_dim=1, time_step=dt, allow_unused=True).to(device)
            stats = None
            if not self.gt:
                path = 'data/run1/pendulum-so3ham-vin-10p-6000.tar'
                model.load_state_dict(torch.load(path, map_location=device))
                #path = './data/run5_dt002_good_retrained_withfixbug/pendulum-so3ham-vin-10p-stats.pkl'
                #stats = from_pickle(path)
            return model, None

        def forward(self, state, action):
            curx = torch.cat((state, action), dim=1)
            if not curx.requires_grad:
                curx.requires_grad = True
            step_num = int(mpc_dt/dt)
            for i in range(step_num):
                if self.gt:
                    nextx = self.model.forward_gt(curx)
                else:
                    nextx = self.model.forward(curx)
                    curx = nextx
            state, _ = torch.split(nextx, [self.model.rotmatdim + self.model.angveldim, self.model.u_dim], dim=1)
            return state


    # def angle_normalize(x):
    #     return (((x + math.pi) % (2 * math.pi)) - math.pi)


    # Init angle and control
    init_angle = 0.001#np.pi/4
    u0 = 0.0

    # Create and reset the pendulum environment to the initialized values.
    env = gym.make('MyPendulum-v1', dt=mpc_dt)
    # Record video
    env = gym.wrappers.Monitor(env, './videos/' + 'pendulum' + '/',
                               force=True)  # , video_callable=lambda x: True, force=True
    env.reset(ori_rep='rotmat')
    env.env.state = np.array([init_angle, u0], dtype=np.float64)
    obs = env.env.get_obs()
    # Get state as input for the neural networks
    # y = np.concatenate((obs, np.array([u0])))
    y = torch.tensor(obs, requires_grad=requires_grad, device=device, dtype=torch.float64).view(1, 12)

    # Desired state
    rd = Rotation.from_euler('xyz', [0.0, 0.0, np.pi])
    rd_matrix = rd.as_matrix()
    R_d = torch.unsqueeze(torch.tensor(rd_matrix, device=device, dtype=torch.float64), dim=0)
    R_d = R_d.view(-1, 9)
    w_d = torch.zeros([1,3], device=device, dtype=torch.float64)
    yd = torch.cat((R_d, w_d), dim = 1)

    nx = 12
    nu = 1

    u_init = None
    render = True
    retrain_after_iter = 100
    run_iter = 200

    # swingup goal (observe theta and theta_dt)
    goal_weights = 2*torch.tensor((1., 1., 1., 1., 1., 1., 1., 1., 1., 0.1, 0.1, 0.1), device=device)  # nx
    goal_state = torch.squeeze(yd) #torch.tensor((0., 0.))  # nx
    ctrl_penalty = 0.0001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(nu, device=device)
    ))  # nx + nu
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(nu, device=device)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

    # run MPC
    total_reward = 0
    state_traj = []
    for i in range(run_iter):
        print(i)
        state = env.get_obs().copy()
        state = torch.tensor(state, device=device, requires_grad=requires_grad).view(1, -1)
        command_start = time.perf_counter()
        # recreate controller using updated u_init (kind of wasteful right?)
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)

        # compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, PendulumDynamicsWrapper())
        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu, device=device)), dim=0)

        elapsed = time.perf_counter() - command_start
        u = action.detach().cpu().numpy()
        # Save states for plotting
        s = env.get_state()
        s = np.concatenate((s, u[0]))
        state_traj.append(s)
        #############################
        s, r, _, _ = env.step(u)

        total_reward += r
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)

        if render:
            env.render()


    logger.info("Total reward %f", total_reward)
    # Plot trajectory
    plot_traj(state_traj, mpc_dt*np.arange(len(state_traj)))
    env.close()
