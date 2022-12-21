
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.spatial.transform import Rotation
from LieFVIN import SO3FVIN, from_pickle, compute_rotation_matrix_from_quaternion
solve_ivp = scipy.integrate.solve_ivp

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
dt = 0.02

def get_model(load = True):
    model = SO3FVIN(device=device, u_dim = 1, time_step = dt).to(device)
    stats = None
    if load:
        path = 'data/run2/pendulum-so3ham-vin-10p-6000.tar'
        model.load_state_dict(torch.load(path, map_location=device))
        path = 'data/run2/pendulum-so3ham-vin-10p-stats.pkl'
        stats = from_pickle(path)
    return model, stats

if __name__ == "__main__":
    savefig = True
    # Figure and font size
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    # Load trained model
    model, stats = get_model()
    # Scale factor for M^-1, V, g neural networks
    beta = 1.0/0.278 #

    # Plot loss
    fig = plt.figure(figsize=figsize, linewidth=5)
    ax = fig.add_subplot(111)
    train_loss = stats['train_loss']
    test_loss = stats['test_loss']
    ax.plot(train_loss[0:6000], 'b', linewidth=line_width, label='train loss')
    ax.plot(test_loss[0:6000], 'r', linewidth=line_width, label='test loss')
    plt.xlabel("iterations", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.yscale('log')
    plt.legend(fontsize=fontsize)
    if savefig:
        plt.savefig('./png/loss_log.pdf', bbox_inches='tight')
    plt.show()

    # Get state q from a range of pendulum angle theta
    theta = np.linspace(-5.0, 5.0, 40)
    q_tensor = torch.tensor(theta, dtype=torch.float64).view(40, 1).to(device)
    q_zeros = torch.zeros(40,2).to(device)
    quat = torch.cat((torch.cos(q_tensor/2), q_zeros, torch.sin(q_tensor/2)), dim=1)
    rotmat = compute_rotation_matrix_from_quaternion(quat)
    # This is the generalized coordinates q = R
    rotmat = rotmat.view(rotmat.shape[0], 9)



    # Calculate the M^-1, V, g for the q.
    M_q_inv = model.M_net(rotmat)
    V_q = model.V_net(rotmat)
    g_q = model.g_net(rotmat)

    # Plot g(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(theta, beta*g_q.detach().cpu().numpy()[:,0], 'b--', linewidth=line_width, label=r'$g(q)[1]$')
    plt.plot(theta, beta * g_q.detach().cpu().numpy()[:, 1], 'r--', linewidth=line_width, label=r'$g(q)[2]$')
    plt.plot(theta, beta * g_q.detach().cpu().numpy()[:, 2], 'g--', linewidth=line_width, label=r'$g(q)[3]$')
    plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(-5, 5)
    plt.ylim(-0.5, 2.5)
    plt.legend(fontsize=fontsize)
    if savefig:
        plt.savefig('./png/g_x.pdf', bbox_inches='tight')
    plt.show()

    # Plot V(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(theta, 5. - 5. * np.cos(theta), 'k--', label='Ground Truth', color='k', linewidth=line_width)
    plt.plot(theta, beta*V_q.detach().cpu().numpy(), 'b', label=r'$U(q)$', linewidth=line_width)
    plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(-5, 5)
    plt.ylim(-8, 12)
    plt.legend(fontsize=fontsize)
    if savefig:
        plt.savefig('./png/V_x.pdf', bbox_inches='tight')
    plt.show()

    # Plot M^-1(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(theta, 3 * np.ones_like(theta), label='Ground Truth', color='k', linewidth=line_width-1)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 2, 2] / beta, 'b--', linewidth=line_width,
             label=r'$J^{-1}(q)[3, 3]$')
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 0, 0] / beta, 'g--', linewidth=line_width,
             label=r'Other $J^{-1}(q)[i,j]$')
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 0, 1] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 0, 2] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 1, 0] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 1, 1] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 1, 2] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 2, 0] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 2, 1] / beta, 'g--', linewidth=line_width)
    plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(-5, 5)
    plt.ylim(-0.5, 6.0)
    plt.legend(fontsize=fontsize)
    if savefig:
        plt.savefig('./png/M_x_all.pdf', bbox_inches='tight')
    plt.show()