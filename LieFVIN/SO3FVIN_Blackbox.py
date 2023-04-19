import torch
import numpy as np
from LieFVIN import MLP, PSD, MatrixNet, hat_map, vee_map, hat_map_batch


class SO3FVIN_BB(torch.nn.Module):
    '''
    Architecture for input (q, q_dot, u),
    where q represent quaternion, a tensor of size (bs, n),
    q and q_dot are tensors of size (bs, n), and
    u is a tensor of size (bs, 1).
    '''

    def __init__(self, device=None, u_dim=3):
        super(SO3FVIN_BB, self).__init__()
        self.rotmatdim = 9
        self.angveldim = 3
        self.u_dim = u_dim
        self.f_net = MLP(self.rotmatdim + self.angveldim + self.u_dim, 1000, self.rotmatdim + self.angveldim, init_gain=0.01).to(device)
        self.device = device
        self.nfe = 0

    def forward(self, x):
        enable_force = True
        #x.requires_grad = True
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            dx = self.f_net(x)
            q_q_dot, uk = torch.split(x, [self.rotmatdim + self.angveldim, self.u_dim], dim=1)

            return torch.cat((dx, uk), dim=1)
    def forward_gt(self, x):
        enable_force = True
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            #zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float64, device=self.device)
            I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            qk, qk_dot, uk = torch.split(x, [self.rotmatdim, self.angveldim, self.u_dim], dim=1)
            Rk = qk.view(-1, 3, 3)
            M_q_inv = 3*I33
            f = np.array([[0.0],
                          [0.0],
                          [1.0]])
            f = torch.tensor(f, dtype=torch.float64).to(self.device)
            f = f.reshape((1, 3))
            g_qk = f.repeat(bs, 1).to(self.device)
            #uk = torch.unsqueeze(uk, dim=2)
            c = 1
            if enable_force:
                if self.u_dim == 1:
                    fk_minus = c*self.h * g_qk * uk
                    fk_plus = (1-c)*self.h * g_qk * uk
                else:
                    fk_minus = c*self.h * torch.squeeze(torch.matmul(g_qk, torch.unsqueeze(uk)))
                    fk_plus = (1-c)*self.h*torch.squeeze(torch.matmul(g_qk, torch.unsqueeze(uk)))
            else:
                fk_minus = torch.zeros(bs, self.angveldim, dtype=torch.float64, device=self.device)
                fk_plus = torch.zeros(bs, self.angveldim, dtype=torch.float64, device=self.device)

            M_q = torch.inverse(M_q_inv)
            #I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            traceM = M_q[:,0,0] + M_q[:,1,1] + M_q[:,2,2]
            traceM = traceM[:, None, None]
            #temp = traceM*I33
            Jd = traceM*I33/2 - M_q
            qk_dot_aug = torch.unsqueeze(qk_dot, dim=2)
            pk = torch.squeeze(torch.matmul(M_q, qk_dot_aug), dim=2)

            # q_p = torch.cat((qk, pk), dim=1)
            # q, p = torch.split(q_p, [self.rotmatdim, self.angveldim], dim=1)
            # M_q_inv = self.M_net(qk)
            # V_q = self.V_net(q)

            V_qk = 5 * (1 - qk[:, 0])
            #tem1 = torch.autograd.grad(V_qk.sum(), qk, create_graph=True, allow_unused=self.allow_unused)[0]
            dVk =  torch.zeros_like(qk)#torch.autograd.grad(V_qk.sum(), qk, create_graph=True, allow_unused=self.allow_unused)[0]
            dVk[:,0] = -5
            dVk = dVk.view(-1, 3, 3)
            SMk = torch.matmul(torch.transpose(dVk, 1, 2), Rk) - torch.matmul(torch.transpose(Rk, 1, 2), dVk)
            Mk = torch.stack((SMk[:, 2, 1], SMk[:, 0, 2], SMk[:, 1, 0]),dim=1)



            a = self.h*pk + self.h**2 * Mk / 2 + self.h *fk_minus
            v = torch.zeros_like(a)
            for i in range(self.implicit_step):
                aTv = torch.unsqueeze(torch.sum(a*v, dim = 1), dim = 1)
                # temp1= torch.cross(a,v, dim=1)
                # temp2 = a*aTv
                # temp3 = 2*torch.squeeze(torch.matmul(Jd, v[:,:,None]))
                phi = a + torch.cross(a,v, dim=1) + v*aTv - 2*torch.squeeze(torch.matmul(M_q, v[:,:,None]))
                # temp1 = hat_map_batch(a)
                # temp2 = aTv[:,:,None]*I33
                dphi = hat_map_batch(a) + aTv[:,:,None]*I33 - 2*M_q + torch.matmul(v[:,:,None], torch.transpose(a[:,:,None], 1,2))
                dphi_inv = torch.inverse(dphi)
                v = v - torch.squeeze(torch.matmul(dphi_inv, phi[:,:,None]))

            Fk0 = torch.matmul((I33 + hat_map_batch(v)), torch.inverse((I33 - hat_map_batch(v))))
            Sv = hat_map_batch(v)
            v = v[:,:,None]
            u2p = 1 + torch.matmul(torch.transpose(v,1,2), v)
            u2m = 1 - torch.matmul(torch.transpose(v, 1, 2), v)
            Fk = (u2m*I33 + 2*Sv + 2*torch.matmul(v, torch.transpose(v, 1, 2)))/u2p
            #Fk = (u2p * I33 + 2 * Sv + 2 * torch.matmul(Sv, Sv)) / u2p
            #+ 2 * torch.matmul(Sv, Sv)
            # R_RT = torch.matmul(Fk, torch.transpose(Fk,1,2))
            # RRT_I = torch.linalg.matrix_norm(R_RT - I33)
            # torch.set_printoptions(precision=16)
            # print("v:", v)
            # print("Error:", RRT_I)

            # v_np = torch.clone(v).detach().cpu().numpy()[0,:,0]
            # Sv_np = hat_map(v_np, mode="numpy")
            # temp1 = (1+np.linalg.norm(v_np)**2)*np.identity(3)
            # temp2 = Sv_np
            # temp3 = np.matmul(Sv_np,Sv_np)
            # temp4 = 1+np.linalg.norm(v_np)**2
            # Fk_np = ((1+np.linalg.norm(v_np)**2)*np.identity(3) + 2*Sv_np + 2*np.matmul(Sv_np,Sv_np))/(1+np.linalg.norm(v_np)**2)
            # R_RT = np.matmul(Fk_np, Fk_np.T)
            # RRT_I = np.linalg.norm(R_RT - np.identity(3))
            # print("v_np:", v_np)
            # print("Error_np:", RRT_I)
            # Fk_np = torch.tensor(Fk_np, dtype=torch.float64, device=self.device)
            # Fk = Fk_np[None,:,:]


            # R_RT = torch.matmul(Fk, torch.transpose(Fk,1,2))
            # RRT_I = torch.linalg.matrix_norm(R_RT - I33)
            # torch.set_printoptions(precision=16)
            # print("Error convert np to torch:", RRT_I)


            #print("Fk[0] = \n", Fk[0,:,:])
            Rk_next = torch.matmul(Rk, Fk)
            qk_next = Rk_next.view(-1, 9)
            V_qk_next = 5 * (1 - qk_next[:, 0])
            #dVk_next =  torch.autograd.grad(V_qk_next.sum(), qk_next, create_graph=True, allow_unused=self.allow_unused)[0]
            dVk_next =  torch.zeros_like(qk_next)#torch.autograd.grad(V_qk.sum(), qk, create_graph=True, allow_unused=self.allow_unused)[0]
            dVk_next[:,0] = -5
            dVk_next = dVk_next.view(-1, 3, 3)
            SMk_next = torch.matmul(torch.transpose(dVk_next, 1, 2), Rk_next) - torch.matmul(torch.transpose(Rk_next, 1, 2), dVk_next)
            Mk_next = torch.stack((SMk_next[:, 2, 1], SMk_next[:, 0, 2], SMk_next[:, 1, 0]), dim = 1)

            FkT = torch.transpose(Fk, 1, 2)
            pk_next = torch.matmul(FkT, pk[:,:,None]) + self.h*torch.matmul(FkT, Mk[:,:,None])/2 + self.h*Mk_next[:,:,None]/2 + torch.matmul(FkT, fk_minus[:,:,None]) + fk_plus[:,:,None]
            dqk_next = torch.matmul(M_q_inv, pk_next)
            dqk_next = dqk_next[:,:,0]

            return torch.cat((qk_next, dqk_next, uk), dim=1)

    def predict(self, step_num, x, gt = False):
        xseq = x[None,:,:]
        curx = x
        for i in range(step_num):
            if gt:
                nextx = self.forward_gt(curx)
            else:
                nextx = self.forward(curx)
            curx = nextx
            xseq = torch.cat((xseq, curx[None,:,:]), dim = 0)

        return xseq
