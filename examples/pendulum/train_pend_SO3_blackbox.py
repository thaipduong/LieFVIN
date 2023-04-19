

import torch, argparse
import numpy as np
import os, sys
import time


THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data/run1'
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from LieFVIN import MLP, PSD
from LieFVIN import SO3FVIN_BB
from data import get_dataset, arrange_data
from LieFVIN import to_pickle, rotmat_L2_geodesic_loss, traj_rotmat_L2_geodesic_loss

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=6000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=100, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pendulum', type=str, help='environment name')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=10,
                        help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--solver', default='vin', type=str, help='type of ODE Solver for Neural ODE')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def train(args):
    dt = 0.02
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the model
    if args.verbose:
        print("Start training with num of points = {} and solver {}.".format(args.num_points, args.solver))
    model = SO3FVIN_BB(device=device, u_dim = 1).to(device)
    # path = './data/pendulum-so3ham-vin-10p-10000.tar'
    # model.load_state_dict(torch.load(path, map_location=device))
    num_parm = get_model_parm_nums(model)
    print('model contains {} parameters'.format(num_parm))
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0.0)

    # Collect data
    us = [0.0, -1., 1., -2., 2.]#, -4.0, 4.0]
    data = get_dataset(seed=args.seed, timesteps=20, save_dir=args.save_dir, us=us, ori_rep="rotmat", dt = dt, samples=128)
    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    test_x, t_eval = arrange_data(data['test_x'], data['t'], num_points=args.num_points)
    train_x_cat = np.concatenate(train_x, axis=1)
    test_x_cat = np.concatenate(test_x, axis=1)
    train_x_cat = torch.tensor(train_x_cat, requires_grad=True, dtype=torch.float64).to(device)
    test_x_cat = torch.tensor(test_x_cat, requires_grad=True, dtype=torch.float64).to(device)
    # train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float64).to(device)
    # test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float64).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float64).to(device)

    # Training stats
    stats = {'train_loss': [], 'test_loss': [], 'forward_time': [], 'backward_time': [], 'nfe': [], 'train_l2_loss': [],\
             'test_l2_loss':[], 'train_geo_loss':[], 'test_geo_loss':[]}

    # Start training
    ground_truth = False
    for step in range(0, args.total_steps + 1):
        train_loss = 0
        test_loss = 0
        train_l2_loss = 0
        train_geo_loss = 0
        test_l2_loss = 0
        test_geo_loss = 0

        t = time.time()
        # Predict states
        train_x_hat = model.predict(args.num_points-1, train_x_cat[0, :, :], gt=ground_truth)
        target = train_x_cat[1:, :, :]
        target_hat = train_x_hat[1:, :, :]
        # Calculate loss
        train_loss_mini, l2_loss_mini, geo_loss_mini = rotmat_L2_geodesic_loss(target, target_hat, split=[model.rotmatdim, model.angveldim, model.u_dim]) #L2_loss(target, target_hat)#
        train_loss = train_loss + train_loss_mini
        train_l2_loss = train_l2_loss + l2_loss_mini
        train_geo_loss = train_geo_loss + geo_loss_mini

        if step % (args.print_every*10) == 0:
            os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
            label = '-bb'
            path = '{}/{}{}-{}-{}p-{}.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points, step)
            torch.save(model.state_dict(), path)

        # Gradient descent
        t = time.time()
        train_loss_mini.backward()
        optim.step()
        optim.zero_grad()
        backward_time = time.time() - t

        # Calculate loss for test data
        test_x_hat = model.predict(args.num_points-1, test_x_cat[0, :, :], gt=ground_truth)
        target = test_x_cat[1:, :, :]
        target_hat = test_x_hat[1:, :, :]
        test_loss_mini, l2_loss_mini, geo_loss_mini = rotmat_L2_geodesic_loss(target, target_hat, split=[model.rotmatdim, model.angveldim, 1])
        test_loss = test_loss + test_loss_mini
        test_l2_loss = test_l2_loss + l2_loss_mini
        test_geo_loss = test_geo_loss + geo_loss_mini

        # Logging stats
        stats['train_loss'].append(train_loss.item())
        stats['test_loss'].append(test_loss.item())
        stats['train_l2_loss'].append(train_l2_loss.item())
        stats['test_l2_loss'].append(test_l2_loss.item())
        stats['train_geo_loss'].append(train_geo_loss.item())
        stats['test_geo_loss'].append(test_geo_loss.item())
        stats['backward_time'].append(backward_time)
        stats['nfe'].append(model.nfe)
        if step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss.item(), test_loss.item()))
            print("step {}, train_l2_loss {:.4e}, test_l2_loss {:.4e}".format(step, train_l2_loss.item(),
                                                                              test_l2_loss.item()))
            print("step {}, train_geo_loss {:.4e}, test_geo_loss {:.4e}".format(step, train_geo_loss.item(),
                                                                                test_geo_loss.item()))
            print("step {}, nfe {:.4e}".format(step, model.nfe))

    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # Save model
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-bb'
    path = '{}/{}{}-{}-{}p.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points)
    torch.save(model.state_dict(), path)
    path = '{}/{}{}-{}-{}p-stats.pkl'.format(args.save_dir, args.name, label, args.solver, args.num_points)
    print("Saved file: ", path)
    to_pickle(stats, path)
