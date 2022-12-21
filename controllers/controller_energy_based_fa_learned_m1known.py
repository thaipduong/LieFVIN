import numpy as np
from controllers.controller_utils import *
import torch, os, sys, argparse
from se3hamneuralode import SE3FAHamNODE_M1, from_pickle

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

class ControllerParams:
    def __init__(self):
        """
        Controller gains and set maximum tilt angle
        """
        #self.maxangle = 40 * np.pi / 180  # you can specify the maximum commanded angle here
        self.K_p = 0.5*np.array([5, 5, 5])  # K_p
        self.K_v = 0.25*np.array([5, 5, 5])  # K_v
        self.K_R = 0.5 * np.array([250, 250, 250])  # K_R
        self.K_w = 0.5 * np.array([20, 20, 20])  # K_w

class LearnedEnergyBasedControllerFA:
    def __init__(self, m, model_path = "data/run5/"):
        self.params = ControllerParams()
        self.m = m
        # self.J = J
        self.grav = 9.8
        self.model_path = model_path
        self.model = self.get_model()


    def get_model(self):
        model = SE3FAHamNODE_M1(device=device, pretrain=False).to(device)
        #path = self.model_path + '/fadronesim-se3ham-rk4-5p-final-5000.tar'
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        #path = self.model_path + '/fadronesim-se3ham-rk4-5p-stats-final-5000.pkl'
        #stats = from_pickle(path)
        return model

    def gen_control(self, qd):
        # Control gain
        K_p = self.params.K_p # K_p
        K_dv = self.params.K_v # K_v
        K_R = self.params.K_R # K_R
        K_dw = self.params.K_w # K_w

        # State
        pos = qd.pos - qd.pos  # y[0:3]
        R = qd.Rot
        RT = R.T
        # Convert velocity to body frame
        v = qd.vel#np.matmul(RT, qd.vel) # Velocity from fas drone is already in the body frame
        w = qd.omega
        y = np.concatenate((pos, R.flatten()))
        pose = torch.tensor(y, requires_grad=True, dtype=torch.float32).to(device)
        pose= pose.view(1,12)
        x_tensor, R_tensor = torch.split(pose, [3, 9], dim=1)

        # Query the values of masses M1, M2, potential energy V, input coeff g.
        g_q = self.model.g_net(pose)
        #V_q = self.model.V_net(pose)
        #M1 = self.model.M_net1(x_tensor)
        M1_inv = np.eye(3) / self.m
        M1 = np.linalg.inv(M1_inv)
        M2 = self.model.M_net2(R_tensor)

        #
        g_pos = g_q.detach().cpu().numpy()[0]
        # g_pos[0:3, 0:3] = np.eye(3)
        # g_pos[0:3, 3:6] = np.zeros((3,3))
        # g_pos[3:6, 0:3] = np.zeros((3,3))
        g_pos_T = g_pos.T
        g_pos_dagger = np.matmul(np.linalg.inv(np.matmul(g_pos_T, g_pos)), g_pos_T)
        #M1_inv = M1.detach().cpu().numpy()[0]
        #M1 = np.linalg.inv(M1_inv)
        M2_inv = M2.detach().cpu().numpy()[0]
        M2 = np.linalg.inv(M2_inv)
        #dVdq = torch.autograd.grad(V_q, pose)[0]
        #dVdq = dVdq.detach().cpu().numpy()[0]
        dVdq = np.array([0, 0, self.m * self.grav, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        w_hat = hat_map(w, mode = "numpy")
        pv = np.matmul(M1, v)
        pw = np.matmul(M2, w)

        # Desired state
        pos_ref = qd.pos_des - qd.pos
        v_ref = qd.vel_des # world frame
        a_ref = qd.acc_des # world frame
        j_ref = np.zeros(3) # world frame
        #yaw_ref = qd.yaw_des
        #yaw_dot_ref = 0

        # Calculate thrust
        RTdV = np.matmul(RT,dVdq[0:3])
        pvxw = np.cross(pv, w)
        RTKp = np.matmul(M1, np.matmul(RT, K_p*(pos - pos_ref)))
        Kdvv_ref =  np.matmul(M1, K_dv*(v - np.matmul(RT, v_ref)))
        pvdot_ref = np.matmul(M1, np.matmul(RT, a_ref) - np.matmul(w_hat, np.matmul(RT, v_ref)))
        b_p = RTdV  - RTKp - Kdvv_ref + pvdot_ref - pvxw# + pvdot_ref # body frame
        #b_p = np.matmul(R, b_p_B) # in world frame


        Rc = qd.rot_des
        wc = qd.omega_des # body frame

        # Calculate b_R
        rxdV = np.cross(R[0,:], dVdq[3:6]) + np.cross(R[1,:], dVdq[6:9]) + np.cross(R[2,:], dVdq[9:12])
        pwxw = np.cross(pw, w)
        pvxv = np.cross(pv, v)
        e_euler = 0.5 * K_R* vee_map(Rc.T @ R - R.T @ Rc)
        kdwwc = K_dw*(w - np.matmul(RT, np.matmul(Rc, wc)))
        b_R = np.matmul(M2, - e_euler - kdwwc) - pwxw - pvxv + rxdV

        # Calculate the control
        wrench = np.hstack((b_p, b_R))
        control = np.matmul(g_pos_dagger, wrench)
        F = control[0:3]#max(0.,control[0])
        M = control[3:]

        return F, M