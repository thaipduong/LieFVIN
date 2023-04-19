import torch, os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import envs
from scipy.spatial.transform import Rotation
from LieFVIN import SO3FVIN_BB, from_pickle


gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

def plot_traj(traj, traj_hat, t_eval):
    ''' Plotting trajectory'''
    fig = plt.figure(figsize=[10, 12])

    plt.subplot(3, 3, 1)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj_hat[:, 0], 'b--', label=r'$\hat R_11$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 2)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj_hat[:, 1], 'b--', label=r'$\hat R_12$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 3)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 2], 'b', label=r'$R_13$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 2], 'b--', label=r'$\hat R_13$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 4)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj_hat[:, 3], 'b--', label=r'$\hat R_21$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 5)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj_hat[:, 4], 'b--', label=r'$\hat R_22$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 6)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 5], 'b', label=r'$R_23$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 5], 'b--', label=r'$\hat R_23$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 7)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 6], 'b', label=r'$R_31$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 6], 'b--', label=r'$\hat R_31$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 8)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 7], 'b', label=r'$R_32$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 7], 'b--', label=r'$\hat R_32$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 9)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 8], 'b', label=r'$R_33$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 8], 'b--', label=r'$\hat R_33$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.tight_layout();
    plt.show()


# Initial condition from gym
import gym
# time intervals
time_step = 2000 ; n_eval = 2000
dt = 0.02
t_span = [0,(time_step-1)*dt]
t_eval = torch.linspace(t_span[0], t_span[1], n_eval)
# init angle
init_angle = np.pi/2
u0 = 0.0



def get_model(load = True):
    model = SO3FVIN_BB(device=device, u_dim = 1).to(device)
    stats = None
    if load:
        path = 'data/run1/pendulum-bb-vin-10p-6000.tar'
        model.load_state_dict(torch.load(path, map_location=device))
    return model, None




######################################### PLOT GROUND-TRUTH ENERGY #########################################

env = gym.make('MyPendulum-v1', dt=dt)
# record video
env = gym.wrappers.Monitor(env, './videos/' + 'pendulum' + '/', force=True)
env=env.env
env.reset(ori_rep='angle', init_angle = init_angle)
#env.state = np.array([init_angle, u0], dtype=np.float32)
obs = env.get_obs()
obs_list = []
obs_list.append(obs)
render = True
for _ in range(time_step-1):
    if render:
        env.render()
    obs, _, _, _ = env.step([u0])
    obs_list.append(obs)

true_ivp = np.stack(obs_list, 1)
true_ivp = np.concatenate((true_ivp, np.zeros((1, time_step))), axis=0)
y0_u_gt = np.asarray([np.cos(init_angle), np.sin(init_angle), 0, u0])
#true_ivp = np.concatenate((true_ivp, np.zeros((1, time_step))), axis=0)
y_gt = true_ivp.T
E_true = true_ivp.T[:, 2]**2 / 6 + 5 * (1 - true_ivp.T[:, 0])
plt.plot(t_eval, E_true, 'b')
plt.show()
env.close()

############################## PLOT TOTAL ENERGY ROLLED OUT FROM OUR DYNAMICS ###############################
savefig = True
angle = []
angle_dot = []

for i in range(len(true_ivp.T)):
    cosy = true_ivp.T[i, 0]
    siny = true_ivp.T[i, 1]
    R = np.array([[cosy, -siny, 0],
                 [siny, cosy, 0],
                 [0, 0, 1]])
    r = Rotation.from_matrix(R)
    a = r.as_euler('zyx')
    angle.append(r.as_euler('zyx')[0])
    angle_dot.append(true_ivp.T[i, 2])


############################## PLOT TOTAL ENERGY ROLLED OUT FROM OUR DYNAMICS ###############################

y0_u = np.array([np.cos(init_angle), -np.sin(init_angle), 0.0, np.sin(init_angle),  np.cos(init_angle), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, u0])#np.concatenate((obs, np.array([u0])))
y0_u = torch.tensor(y0_u, requires_grad=True, device=device, dtype=torch.float64).view(1, 13)

# Roll out our dynamics model from the initial state
model, stats = get_model(load=True)
y = model.predict(len(t_eval)-1, y0_u, gt=False)
y = y.detach().cpu().numpy()
cos_y = y[:,0,0]
sin_y = y[:,0,3]
y_dot = y[:,0,11]
y = y[:,0,:]


# Determine the scaling factor beta and the potential_energy_offset from analyze_pend_SO3.py's results.
# This should be changed according to analyze_pend_SO3.py if we have new results.
total_energy_learned = 0.5*(1/3)*y_dot**2 + + 5 * (1 - cos_y)
# for i in range(len(t_eval)):
#     m = 1/3#np.linalg.inv(M_q_inv[i,:,:])[2,2]
#     energy = 0.5*m*y_dot[i]**2
#     energy = energy + 5 * (1 - cos_y) # scaled and offset
#     total_energy_learned.append(energy)

fig = plt.figure(figsize=(12,7))
plt.plot(t_eval, E_true, 'b', linewidth=4, label='ground truth')
plt.plot(t_eval, total_energy_learned, 'r', linewidth=4, label='learned total energy')
plt.xlabel("$t$", fontsize=24)
plt.ylim(3.5, 6.5)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
if savefig:
    plt.savefig('./png/hamiltonian.pdf', bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(12,7))
plt.plot(t_eval, y_gt[:,0], 'b', linewidth=4, label='ground truth cos')
plt.plot(t_eval, cos_y, 'b--', linewidth=4, label='predicted cos')

plt.xlabel("$t$", fontsize=24)
#plt.ylim(4, 6)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
# if savefig:
#     plt.savefig('./png/hamiltonian.pdf', bbox_inches='tight')
plt.show()

############################## PLOT SE(3) CONSTRAINTS ROLLED OUT FROM OUR DYNAMICS ###############################

det = []
RRT_I_dist = []
for i in range(len(y)):
    R_hat = y[i, 0:9]
    R_hat = R_hat.reshape(3, 3)
    R_det = np.linalg.det(R_hat)
    det.append(np.abs(R_det - 1))
    R_RT = np.matmul(R_hat, R_hat.transpose())
    RRT_I = np.linalg.norm(R_RT - np.diag([1.0, 1.0, 1.0]))
    RRT_I_dist.append(RRT_I)

figsize = (12, 7.8)
fontsize = 24
fontsize_ticks = 32
line_width = 4
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
ax.plot(t_eval, det, 'b', linewidth=line_width, label=r'$|det(R) - 1|$')
ax.plot(t_eval, RRT_I_dist, 'r', linewidth=line_width, label=r'$\Vert R R^\top - I\Vert$')
plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.legend(fontsize=fontsize)
if savefig:
    plt.savefig('./png/SO3_constraints.pdf', bbox_inches='tight')
plt.show()

############################## PLOT PHASE PORTRAIT ROLLED OUT FROM OUR DYNAMICS ###############################

fig = plt.figure(figsize=figsize, linewidth=5)
ax = fig.add_subplot(111)
angle_hat = []
angle_dot_hat = []
for i in range(len(y)):
    R_hat = y[i, 0:9]
    R_hat = R_hat.reshape(3, 3)
    r = Rotation.from_matrix(R_hat)
    angle_hat.append(r.as_euler('zyx')[0])
    angle_dot_hat.append(y[i, 11])

ax.plot(angle, angle_dot, 'b', linewidth=line_width*2, label='Ground truth')
ax.plot(angle_hat,angle_dot_hat, 'r--', linewidth=line_width, label='Our trajectory')
plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
plt.ylabel("angular velocity", fontsize=fontsize_ticks)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.legend(fontsize=fontsize)
if savefig:
    plt.savefig('./png/phase_portrait.pdf', bbox_inches='tight')
plt.show()

np.savez("./data/run1/rollout_bb_data.npz", t_eval=t_eval, E_true=E_true, \
         total_energy_learned=total_energy_learned, det=det, RRT_I_dist=RRT_I_dist,\
         angle=angle, angle_dot=angle_dot, angle_hat=angle_hat, angle_dot_hat=angle_dot_hat,\
         )