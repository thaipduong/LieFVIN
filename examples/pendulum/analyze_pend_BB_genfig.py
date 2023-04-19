
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.spatial.transform import Rotation
from LieFVIN import SO3FVIN_BB, from_pickle, compute_rotation_matrix_from_quaternion
solve_ivp = scipy.integrate.solve_ivp

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
dt = 0.02

def get_model(load = True):
    model = SO3FVIN_BB(device=device, u_dim = 1).to(device)
    stats = None
    if load:
        path = 'data/run1/pendulum-bb-vin-10p-6000.tar'
        model.load_state_dict(torch.load(path, map_location=device))
        path = 'data/run1/pendulum-bb-vin-10p-stats.pkl'
        stats = from_pickle(path)
    return model, stats

if __name__ == "__main__":
    savefig = False
    # Figure and font size
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    # Load trained model
    model, stats = get_model()
    # Scale factor for M^-1, V, g neural networks

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

    np.savez("./data/run1/loss_bb.npz", train_loss=train_loss[0:6000], test_loss=test_loss[0:6000])