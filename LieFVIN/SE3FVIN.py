
import torch
import numpy as np
from LieFVIN import MLP, PSD, MatrixNet, Mass, hat_map, vee_map, hat_map_batch, MassFixed, vee_map_batch

#from LieFVIN import MLP, PSD, MatrixNet
from LieFVIN import compute_rotation_matrix_from_quaternion
from .utils import L2_loss



class SE3FVIN(torch.nn.Module):
    def __init__(self, device=None, M_net1  = None, M_net2 = None, V_net = None, dV_net = None, g_net = None, udim = 4, time_step = 0.01, pretrain = False):
        super(SE3FVIN, self).__init__()
        init_gain = 1.0
        self.device = device
        self.xdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.posedim = self.xdim + self.Rdim #3 for position + 12 for rotmat
        self.twistdim = self.linveldim + self.angveldim #3 for linear vel + 3 for ang vel
        self.udim = udim
        if M_net1 is None:
            #self.M_net1 = PSD(self.xdim, 200, self.linveldim, init_gain=init_gain).to(device)
            eps = torch.Tensor([10., 10., 10.]).to(self.device)#torch.Tensor([37., 37., 37.]).to(self.device)
            #self.M_net1 = Mass(m_dim=3, eps=eps, init_gain=1).to(device)
            self.M_net1 = MassFixed(m_dim=3, eps=eps, init_gain=1).to(device)
        else:
            self.M_net1 = M_net1
        if M_net2 is None:
            #self.M_net2 = PSD(self.Rdim, 400, self.twistdim - self.linveldim, init_gain=0.1*init_gain).to(device)
            eps = torch.Tensor([100., 100., 100.]).to(self.device)#torch.Tensor(np.array([1 / 2.3951, 1 / 2.3951, 1 / 3.2347]) * 1e5).to(self.device)
            self.M_net2 = Mass(m_dim=3, eps=eps, init_gain=1).to(device)
        else:
            self.M_net2 = M_net2
        if V_net is None:
            self.V_net = MLP(self.posedim, 10, 1, init_gain=init_gain).to(device)
        else:
            self.V_net = V_net
        if dV_net is None:
            self.dV_net = MLP(self.posedim, 10, self.posedim, init_gain=init_gain).to(device)
        else:
            self.dV_net = dV_net
        if g_net is None:
            self.g_net = MatrixNet(self.posedim, 10, self.twistdim*self.udim, shape=(self.twistdim,self.udim), init_gain=3*init_gain).to(device)
        else:
            self.g_net = g_net
        self.nfe = 0
        self.G = 9.8
        self.h = time_step
        self.implicit_step = 3
        if pretrain:
            self.pretrain()

    def pretrain(self):
        x = np.arange(-10, 10, 0.5)
        y = np.arange(-10, 10, 0.5)
        z = np.arange(-10, 10, 0.5)
        n_grid = len(z)
        batch = n_grid ** 3
        xx, yy, zz = np.meshgrid(x, y, z)
        Xgrid = np.zeros([batch, 3])
        Xgrid[:, 0] = np.reshape(xx, (batch,))
        Xgrid[:, 1] = np.reshape(yy, (batch,))
        Xgrid[:, 2] = np.reshape(zz, (batch,))
        Xgrid = torch.tensor(Xgrid, dtype=torch.float64).view(batch, 3).to(self.device)
        ######################################
        m_net1_hat = self.M_net1(Xgrid)
        m = 1.0
        m_guess = m*torch.eye(3)
        m_guess = m_guess.reshape((1, 3, 3))
        m_guess = m_guess.repeat(batch, 1, 1).to(self.device)
        optim1 = torch.optim.Adam(self.M_net1.parameters(), 1e-3, weight_decay=0.0)
        loss = L2_loss(m_net1_hat, m_guess)
        print("Start pretraining Mnet1!", loss.detach().cpu().numpy())
        step = 0
        max_iter = 1000
        while loss > 1e-6 and step < max_iter:
            loss.backward()
            optim1.step()
            optim1.zero_grad()
            if step%10 == 0:
                print("step", step, loss.detach().cpu().numpy())
            m_net1_hat = self.M_net1(Xgrid)
            loss = L2_loss(m_net1_hat, m_guess)
            step = step + 1
        print("Pretraining Mnet1 done!", loss.detach().cpu().numpy())
        ######################################
        del Xgrid
        torch.cuda.empty_cache()

        batch = 50000
        # Uniformly generate quaternion using http://planning.cs.uiuc.edu/node198.html
        rand_ =np.random.uniform(size=(batch, 3))
        u1, u2, u3 = rand_[:,0], rand_[:, 1], rand_[:, 2]
        quat = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                              np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])
        q_tensor = torch.tensor(quat.transpose(), dtype=torch.float64).view(batch, 4).to(self.device)
        R_tensor = compute_rotation_matrix_from_quaternion(q_tensor)
        R_tensor = R_tensor.view(-1, 9)
        m_net2_hat = self.M_net2(R_tensor)
        inertia_guess = 1*torch.eye(3)
        inertia_guess[2,2] = 1
        #inertia_guess = 71429*torch.eye(3)
        #inertia_guess[2:2] = 46083
        inertia_guess = inertia_guess.reshape((1, 3, 3))
        inertia_guess = inertia_guess.repeat(batch, 1, 1).to(self.device)
        optim = torch.optim.Adam(self.M_net2.parameters(), 1e-3, weight_decay=0.0)
        loss = L2_loss(m_net2_hat, inertia_guess)
        print("Start pretraining Mnet2!", loss.detach().cpu().numpy())
        step = 0
        while loss > 1e-6 and step < max_iter:
            loss.backward()
            optim.step()
            optim.zero_grad()
            if step%10 == 0:
                print("step", step, loss.detach().cpu().numpy())
            m_net2_hat = self.M_net2(R_tensor)
            loss = L2_loss(m_net2_hat, inertia_guess)
            step = step + 1
        print("Pretraining Mnet2 done!", loss.detach().cpu().numpy())
        del q_tensor
        torch.cuda.empty_cache()


    def forward_traininga(self, x):
        enable_force = True
        gtM, gtG, gtV = False, False, False
        use_dVNet = False
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            #zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float64, device=self.device)
            I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            qk, qk_dot, uk = torch.split(x, [self.posedim, self.twistdim, self.udim], dim=1)

            qxk, qRk = torch.split(qk, [self.xdim, self.Rdim], dim=1)
            vk, omegak = torch.split(qk_dot, [self.linveldim, self.angveldim], dim=1)
            Rk = qRk.view(-1, 3, 3)

            m = 0.027
            if gtM:
                Mx_inv = (1 / m) * I33
                J_inv = np.diag([1 / 2.3951, 1 / 2.3951, 1 / 3.2347]) * 1e5  # np.diag([1/1.4, 1/1.4, 1/2.17])*1e5
                J_inv = torch.tensor(J_inv, dtype=torch.float64).to(self.device)
                J_inv = J_inv.reshape((1, 3, 3))
                MR_inv = J_inv.repeat(bs, 1, 1).to(self.device)
            else:
                Mx_inv = self.M_net1(qxk)
                MR_inv = self.M_net2(qRk)

            if gtG:
                f = np.array([[0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
                f = torch.tensor(f, dtype=torch.float64).to(self.device)
                f = f.reshape((1, 6, 4))
                g_qk = f.repeat(bs, 1, 1).to(self.device)
            else:
                g_qk = self.g_net(qk)
                #print("##################: \n", g_qk[0,:,:])

            #uk = torch.unsqueeze(uk, dim=2)
            c = 0.5
            if enable_force:
                if self.udim == 1:
                    fk_minus = c*self.h * g_qk * uk
                    fk_plus = (1-c)*self.h * g_qk * uk
                else:
                    fk_minus = c*self.h *torch.matmul(g_qk, torch.unsqueeze(uk, dim = 2))
                    fk_plus = (1-c)*self.h*torch.matmul(g_qk, torch.unsqueeze(uk, dim = 2))
            else:
                # Not sure if this is the right ground-truth?
                fk_minus = torch.zeros(bs, self.twistdim, 1 , dtype=torch.float64, device=self.device)
                fk_plus = torch.zeros(bs, self.twistdim, 1, dtype=torch.float64, device=self.device)

            fxk_minus, fRk_minus = torch.split(fk_minus, [self.linveldim, self.angveldim], dim=1)
            fxk_plus, fRk_plus = torch.split(fk_plus, [self.linveldim, self.angveldim], dim=1)
            MR = torch.inverse(MR_inv)
            Mx = torch.inverse(Mx_inv)
            #I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            traceM = MR[:,0,0] + MR[:,1,1] + MR[:,2,2]
            traceM = traceM[:, None, None]
            #temp = traceM*I33
            Jd = traceM*I33/2 - MR
            omegak_aug = torch.unsqueeze(omegak, dim=2)
            pRk = torch.squeeze(torch.matmul(MR, omegak_aug), dim=2)
            vk_aug = torch.unsqueeze(vk, dim=2)
            pxk = torch.squeeze(torch.matmul(Mx, vk_aug), dim=2)

            if use_dVNet:
                dVqk = self.dV_net(qk)
            else:
                if gtV:
                    V_qk = m * self.G * qk[:, 2]
                else:
                    V_qk = self.V_net(qk)
                dVqk = torch.autograd.grad(V_qk.sum(), qk, create_graph=True)[0]
            dVxk, dVRk = torch.split(dVqk, [self.xdim, self.Rdim], dim=1)
            dVRk = dVRk.view(-1, 3, 3)
            SMk = torch.matmul(torch.transpose(dVRk, 1, 2), Rk) - torch.matmul(torch.transpose(Rk, 1, 2), dVRk)
            Mk = torch.stack((SMk[:, 2, 1], SMk[:, 0, 2], SMk[:, 1, 0]),dim=1)


            alpha = 0.5
            a = self.h*pRk + (1-alpha)*self.h**2 * Mk + self.h *torch.squeeze(fRk_minus)
            v = torch.zeros_like(a)
            for i in range(self.implicit_step):
                aTv = torch.unsqueeze(torch.sum(a*v, dim = 1), dim = 1)
                # temp1= torch.cross(a,v, dim=1)
                # temp2 = a*aTv
                # temp3 = 2*torch.squeeze(torch.matmul(Jd, v[:,:,None]))
                phi = a + torch.cross(a,v, dim=1) + v*aTv - \
                      2*torch.squeeze(torch.matmul(MR, v[:,:,None]))
                # temp1 = hat_map_batch(a)
                # temp2 = aTv[:,:,None]*I33
                dphi = hat_map_batch(a) + aTv[:,:,None]*I33 - 2*MR + torch.matmul(v[:,:,None], torch.transpose(a[:,:,None], 1,2))
                dphi_inv = torch.inverse(dphi)
                v = v - torch.squeeze(torch.matmul(dphi_inv, phi[:,:,None]))

            #Fk0 = torch.matmul((I33 + hat_map_batch(v)), torch.inverse((I33 - hat_map_batch(v))))
            Sv = hat_map_batch(v)
            v = v[:,:,None]
            u2 = 1 + torch.matmul(torch.transpose(v,1,2), v)
            Fk = (u2*I33 + 2*Sv + 2 * torch.matmul(Sv, Sv))/u2

            Rk_next = torch.matmul(Rk, Fk)
            qRk_next = Rk_next.view(-1, 9)
            qxk_next = qxk + self.h*torch.squeeze(torch.matmul(Mx_inv, torch.unsqueeze(pxk, dim=2))) - \
                       self.h*torch.squeeze(torch.matmul(Mx_inv, torch.matmul(Rk, fxk_minus)))  - \
                       ((1-alpha)*(self.h**2))*torch.squeeze(torch.matmul(Mx_inv,torch.unsqueeze(dVxk,dim=2)))
            qk_next = torch.cat((qxk_next, qRk_next), dim = 1)

            if use_dVNet:
                dVqk_next = self.dV_net(qk_next)
            else:
                if gtV:
                    V_qk_next = m * self.G * qk_next[:, 2]
                else:
                    V_qk_next = self.V_net(qk_next)
                dVqk_next = torch.autograd.grad(V_qk_next.sum(), qk_next, create_graph=True)[0]

            dVxk_next, dVRk_next = torch.split(dVqk_next, [self.xdim, self.Rdim], dim=1)
            dVRk_next = dVRk_next.view(-1, 3, 3)
            SMk_next = torch.matmul(torch.transpose(dVRk_next, 1, 2), Rk_next) - \
                       torch.matmul(torch.transpose(Rk_next, 1, 2), dVRk_next)
            Mk_next = torch.stack((SMk_next[:, 2, 1], SMk_next[:, 0, 2], SMk_next[:, 1, 0]), dim = 1)

            FkT = torch.transpose(Fk, 1, 2)
            pRk_next = torch.matmul(FkT, pRk[:,:,None]) + (1-alpha)*self.h*torch.matmul(FkT, Mk[:,:,None]) +\
                       alpha*self.h*Mk_next[:,:,None] + torch.matmul(FkT, fRk_minus) + fRk_plus

            pxk_next = -(1-alpha)*self.h*dVxk - alpha*self.h*dVxk_next + \
                       torch.squeeze(torch.matmul(Rk, fxk_minus)) + torch.squeeze(torch.matmul(Rk_next, fxk_plus))
            omegak_next = torch.matmul(MR_inv, pRk_next)
            omegak_next = omegak_next[:,:,0]
            vk_next = torch.matmul(Mx_inv, torch.unsqueeze(pxk_next, dim = 2)) + vk_aug
            vk_next = vk_next[:,:,0]

            return torch.cat((qk_next, vk_next, omegak_next, uk), dim=1)

    def forward_trainingb(self, x, x_next_data):
        enable_force = True
        gtM, gtG, gtV = False, False, False
        use_dVNet = False
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            #zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float64, device=self.device)
            I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            qk, qk_dot, uk = torch.split(x, [self.posedim, self.twistdim, self.udim], dim=1)

            qxk, qRk = torch.split(qk, [self.xdim, self.Rdim], dim=1)

            vk, omegak = torch.split(qk_dot, [self.linveldim, self.angveldim], dim=1)
            Rk = qRk.view(-1, 3, 3)

            m = 0.027
            if gtM:
                Mx_inv = (1 / m) * I33
                J_inv = np.diag([1 / 2.3951, 1 / 2.3951, 1 / 3.2347]) * 1e5  # np.diag([1/1.4, 1/1.4, 1/2.17])*1e5
                J_inv = torch.tensor(J_inv, dtype=torch.float64).to(self.device)
                J_inv = J_inv.reshape((1, 3, 3))
                MR_inv = J_inv.repeat(bs, 1, 1).to(self.device)
            else:
                Mx_inv = self.M_net1(qxk)
                MR_inv = self.M_net2(qRk)

            if gtG:
                f = np.array([[0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
                f = torch.tensor(f, dtype=torch.float64).to(self.device)
                f = f.reshape((1, 6, 4))
                g_qk = f.repeat(bs, 1, 1).to(self.device)
            else:
                g_qk = self.g_net(qk)
                #print("##################: \n", g_qk[0,:,:])

            #uk = torch.unsqueeze(uk, dim=2)
            c = 0.5
            if enable_force:
                if self.udim == 1:
                    fk_minus = c*self.h * g_qk * uk
                    fk_plus = (1-c)*self.h * g_qk * uk
                else:
                    fk_minus = c*self.h *torch.matmul(g_qk, torch.unsqueeze(uk, dim = 2))
                    fk_plus = (1-c)*self.h*torch.matmul(g_qk, torch.unsqueeze(uk, dim = 2))
            else:
                # Not sure if this is the right ground-truth?
                fk_minus = torch.zeros(bs, self.twistdim, 1 , dtype=torch.float64, device=self.device)
                fk_plus = torch.zeros(bs, self.twistdim, 1, dtype=torch.float64, device=self.device)

            fxk_minus, fRk_minus = torch.split(fk_minus, [self.linveldim, self.angveldim], dim=1)
            fxk_plus, fRk_plus = torch.split(fk_plus, [self.linveldim, self.angveldim], dim=1)
            MR = torch.inverse(MR_inv)
            Mx = torch.inverse(Mx_inv)
            #I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            traceM = MR[:,0,0] + MR[:,1,1] + MR[:,2,2]
            traceM = traceM[:, None, None]
            #temp = traceM*I33
            Jd = traceM*I33/2 - MR
            omegak_aug = torch.unsqueeze(omegak, dim=2)
            pRk = torch.squeeze(torch.matmul(MR, omegak_aug), dim=2)
            vk_aug = torch.unsqueeze(vk, dim=2)
            pxk = torch.squeeze(torch.matmul(Mx, vk_aug), dim=2)

            if use_dVNet:
                dVqk = self.dV_net(qk)
            else:
                if gtV:
                    V_qk = m * self.G * qk[:, 2]
                else:
                    V_qk = self.V_net(qk)
                dVqk = torch.autograd.grad(V_qk.sum(), qk, create_graph=True)[0]
            dVxk, dVRk = torch.split(dVqk, [self.xdim, self.Rdim], dim=1)
            dVRk = dVRk.view(-1, 3, 3)
            SMk = torch.matmul(torch.transpose(dVRk, 1, 2), Rk) - torch.matmul(torch.transpose(Rk, 1, 2), dVRk)
            Mk = torch.stack((SMk[:, 2, 1], SMk[:, 0, 2], SMk[:, 1, 0]),dim=1)


            alpha = 0.5
            a = self.h*pRk + (1-alpha)*self.h**2 * Mk + self.h *torch.squeeze(fRk_minus)

            #####################################################################################
            qk_next_data, _, _ = torch.split(x_next_data, [self.posedim, self.twistdim, self.udim], dim=1)

            _, qRk_next_data = torch.split(qk_next_data, [self.xdim, self.Rdim], dim=1)
            Rk_next_data = qRk_next_data.view(-1, 3, 3)
            Fk = torch.matmul(torch.transpose(Rk, 1, 2), Rk_next_data)

            Sv = torch.matmul(Fk, Jd) - torch.matmul(Jd, torch.transpose(Fk, 1, 2))
            v = vee_map_batch(Sv)

            implicit_eq_loss = L2_loss(a, v)
            #####################################################################################
            Rk_next = torch.matmul(Rk, Fk)
            qRk_next = Rk_next.view(-1, 9)
            qxk_next = qxk + self.h*torch.squeeze(torch.matmul(Mx_inv, torch.unsqueeze(pxk, dim=2))) - \
                       self.h*torch.squeeze(torch.matmul(Mx_inv, torch.matmul(Rk, fxk_minus)))  - \
                       ((1-alpha)*(self.h**2))*torch.squeeze(torch.matmul(Mx_inv,torch.unsqueeze(dVxk,dim=2)))
            qk_next = torch.cat((qxk_next, qRk_next), dim = 1)

            if use_dVNet:
                dVqk_next = self.dV_net(qk_next)
            else:
                if gtV:
                    V_qk_next = m * self.G * qk_next[:, 2]
                else:
                    V_qk_next = self.V_net(qk_next)
                dVqk_next = torch.autograd.grad(V_qk_next.sum(), qk_next, create_graph=True)[0]

            dVxk_next, dVRk_next = torch.split(dVqk_next, [self.xdim, self.Rdim], dim=1)
            dVRk_next = dVRk_next.view(-1, 3, 3)
            SMk_next = torch.matmul(torch.transpose(dVRk_next, 1, 2), Rk_next) - \
                       torch.matmul(torch.transpose(Rk_next, 1, 2), dVRk_next)
            Mk_next = torch.stack((SMk_next[:, 2, 1], SMk_next[:, 0, 2], SMk_next[:, 1, 0]), dim = 1)

            FkT = torch.transpose(Fk, 1, 2)
            pRk_next = torch.matmul(FkT, pRk[:,:,None]) + (1-alpha)*self.h*torch.matmul(FkT, Mk[:,:,None]) +\
                       alpha*self.h*Mk_next[:,:,None] + torch.matmul(FkT, fRk_minus) + fRk_plus
            pxk_next = -(1-alpha)*self.h*dVxk - alpha*self.h*dVxk_next + \
                       torch.squeeze(torch.matmul(Rk, fxk_minus)) + torch.squeeze(torch.matmul(Rk_next, fxk_plus))
            omegak_next = torch.matmul(MR_inv, pRk_next)
            omegak_next = omegak_next[:,:,0]
            vk_next = torch.matmul(Mx_inv, torch.unsqueeze(pxk_next, dim = 2)) + vk_aug
            vk_next = vk_next[:,:,0]

            return torch.cat((qk_next, vk_next, omegak_next, uk), dim=1), implicit_eq_loss

    def forward_gt(self, x):
        enable_force = True
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            #zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float64, device=self.device)
            I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            qk, qk_dot, uk = torch.split(x, [self.posedim, self.twistdim, self.udim], dim=1)

            qxk, qRk = torch.split(qk, [self.xdim, self.Rdim], dim=1)
            vk, omegak = torch.split(qk_dot, [self.linveldim, self.angveldim], dim=1)
            Rk = qRk.view(-1, 3, 3)
            m = 0.027
            Mx_inv = (1/m)*I33
            #Mx_inv = self.M_net1(qxk)

            J_inv = np.diag([1/2.3951, 1/2.3951, 1/3.2347])*1e5 #np.diag([1/1.4, 1/1.4, 1/2.17])*1e5
            J_inv = torch.tensor(J_inv, dtype=torch.float64).to(self.device)
            J_inv = J_inv.reshape((1, 3, 3))
            MR_inv = J_inv.repeat(bs, 1, 1).to(self.device)


            f = np.array([[0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
            f = torch.tensor(f, dtype=torch.float64).to(self.device)
            f = f.reshape((1, 6, 4))
            g_qk = f.repeat(bs, 1, 1).to(self.device)
            #uk = torch.unsqueeze(uk, dim=2)
            c = 0.5
            if enable_force:
                if self.udim == 1:
                    fk_minus = c*self.h * g_qk * uk
                    fk_plus = (1-c)*self.h * g_qk * uk
                else:
                    fk_minus = c*self.h *torch.matmul(g_qk, torch.unsqueeze(uk, dim = 2))
                    fk_plus = (1-c)*self.h*torch.matmul(g_qk, torch.unsqueeze(uk, dim = 2))
            else:
                # Not sure if this is the right ground-truth?
                fk_minus = torch.zeros(bs, self.twistdim, 1 , dtype=torch.float64, device=self.device)
                fk_plus = torch.zeros(bs, self.twistdim, 1, dtype=torch.float64, device=self.device)

            fxk_minus, fRk_minus = torch.split(fk_minus, [self.linveldim, self.angveldim], dim=1)
            fxk_plus, fRk_plus = torch.split(fk_plus, [self.linveldim, self.angveldim], dim=1)
            MR = torch.inverse(MR_inv)
            Mx = torch.inverse(Mx_inv)
            #I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            traceM = MR[:,0,0] + MR[:,1,1] + MR[:,2,2]
            traceM = traceM[:, None, None]
            #temp = traceM*I33
            Jd = traceM*I33/2 - MR
            omegak_aug = torch.unsqueeze(omegak, dim=2)
            pRk = torch.squeeze(torch.matmul(MR, omegak_aug), dim=2)
            vk_aug = torch.unsqueeze(vk, dim=2)
            pxk = torch.squeeze(torch.matmul(Mx, vk_aug), dim=2)

            V_qk = m*self.G*qk[:,2]
            dVxRk =  torch.autograd.grad(V_qk.sum(), qk, create_graph=True)[0]
            dVxk, dVRk = torch.split(dVxRk, [self.xdim, self.Rdim], dim=1)
            dVRk = dVRk.view(-1, 3, 3)
            SMk = torch.matmul(torch.transpose(dVRk, 1, 2), Rk) - torch.matmul(torch.transpose(Rk, 1, 2), dVRk)
            Mk = torch.stack((SMk[:, 2, 1], SMk[:, 0, 2], SMk[:, 1, 0]),dim=1)


            alpha = 0.5
            a = self.h*pRk + (1-alpha)*self.h**2 * Mk + self.h *torch.squeeze(fRk_minus)
            v = torch.zeros_like(a)
            for i in range(self.implicit_step):
                aTv = torch.unsqueeze(torch.sum(a*v, dim = 1), dim = 1)

                phi = a + torch.cross(a,v, dim=1) + v*aTv - \
                      2*torch.squeeze(torch.matmul(MR, v[:,:,None]))

                dphi = hat_map_batch(a) + aTv[:,:,None]*I33 - 2*MR + torch.matmul(v[:,:,None], torch.transpose(a[:,:,None], 1,2))
                dphi_inv = torch.inverse(dphi)
                v = v - torch.squeeze(torch.matmul(dphi_inv, phi[:,:,None]))


            Sv = hat_map_batch(v)
            v = v[:,:,None]
            u2 = 1 + torch.matmul(torch.transpose(v,1,2), v)
            Fk = (u2*I33 + 2*Sv + 2 * torch.matmul(Sv, Sv))/u2

            Rk_next = torch.matmul(Rk, Fk)
            qRk_next = Rk_next.view(-1, 9)
            qxk_next = qxk + self.h*torch.squeeze(torch.matmul(Mx_inv, torch.unsqueeze(pxk, dim=2))) - \
                       self.h*torch.squeeze(torch.matmul(Mx_inv, torch.matmul(Rk, fxk_minus)))  - \
                       ((1-alpha)*(self.h**2))*torch.squeeze(torch.matmul(Mx_inv,torch.unsqueeze(dVxk,dim=2)))
            qk_next = torch.cat((qxk_next, qRk_next), dim = 1)

            V_qk_next = m*self.G*qk_next[:,2]
            dVqk_next =  torch.autograd.grad(V_qk_next.sum(), qk_next, create_graph=True)[0]
            dVxk_next, dVRk_next = torch.split(dVqk_next, [self.xdim, self.Rdim], dim=1)
            dVRk_next = dVRk_next.view(-1, 3, 3)
            SMk_next = torch.matmul(torch.transpose(dVRk_next, 1, 2), Rk_next) - \
                       torch.matmul(torch.transpose(Rk_next, 1, 2), dVRk_next)
            Mk_next = torch.stack((SMk_next[:, 2, 1], SMk_next[:, 0, 2], SMk_next[:, 1, 0]), dim = 1)

            FkT = torch.transpose(Fk, 1, 2)
            pRk_next = torch.matmul(FkT, pRk[:,:,None]) + (1-alpha)*self.h*torch.matmul(FkT, Mk[:,:,None]) +\
                       alpha*self.h*Mk_next[:,:,None] + torch.matmul(FkT, fRk_minus) + fRk_plus
            pxk_next = pxk - (1-alpha)*self.h*dVxk - alpha*self.h*dVxk_next + \
                       torch.squeeze(torch.matmul(Rk, fxk_minus)) + torch.squeeze(torch.matmul(Rk_next, fxk_plus))
            omegak_next = torch.matmul(MR_inv, pRk_next)
            omegak_next = omegak_next[:,:,0]
            vk_next = torch.matmul(Mx_inv, torch.unsqueeze(pxk_next, dim = 2))
            vk_next = vk_next[:,:,0]

            return torch.cat((qk_next, vk_next, omegak_next, uk), dim=1)


    def forward_gt_trainingb(self, x, x_next_data):
        enable_force = True
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            #zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float64, device=self.device)
            I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            qk, qk_dot, uk = torch.split(x, [self.posedim, self.twistdim, self.udim], dim=1)

            qxk, qRk = torch.split(qk, [self.xdim, self.Rdim], dim=1)
            vk, omegak = torch.split(qk_dot, [self.linveldim, self.angveldim], dim=1)
            Rk = qRk.view(-1, 3, 3)
            m = 0.027
            Mx_inv = (1/m)*I33
            #Mx_inv = self.M_net1(qxk)

            J_inv = np.diag([1/2.3951, 1/2.3951, 1/3.2347])*1e5 #np.diag([1/1.4, 1/1.4, 1/2.17])*1e5
            J_inv = torch.tensor(J_inv, dtype=torch.float64).to(self.device)
            J_inv = J_inv.reshape((1, 3, 3))
            MR_inv = J_inv.repeat(bs, 1, 1).to(self.device)


            f = np.array([[0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
            f = torch.tensor(f, dtype=torch.float64).to(self.device)
            f = f.reshape((1, 6, 4))
            g_qk = f.repeat(bs, 1, 1).to(self.device)
            #uk = torch.unsqueeze(uk, dim=2)
            c = 0.5
            if enable_force:
                if self.udim == 1:
                    fk_minus = c*self.h * g_qk * uk
                    fk_plus = (1-c)*self.h * g_qk * uk
                else:
                    fk_minus = c*self.h *torch.matmul(g_qk, torch.unsqueeze(uk, dim = 2))
                    fk_plus = (1-c)*self.h*torch.matmul(g_qk, torch.unsqueeze(uk, dim = 2))
            else:
                # Not sure if this is the right ground-truth?
                fk_minus = torch.zeros(bs, self.twistdim, 1 , dtype=torch.float64, device=self.device)
                fk_plus = torch.zeros(bs, self.twistdim, 1, dtype=torch.float64, device=self.device)

            fxk_minus, fRk_minus = torch.split(fk_minus, [self.linveldim, self.angveldim], dim=1)
            fxk_plus, fRk_plus = torch.split(fk_plus, [self.linveldim, self.angveldim], dim=1)
            MR = torch.inverse(MR_inv)
            Mx = torch.inverse(Mx_inv)
            #I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            traceM = MR[:,0,0] + MR[:,1,1] + MR[:,2,2]
            traceM = traceM[:, None, None]
            #temp = traceM*I33
            Jd = traceM*I33/2 - MR
            omegak_aug = torch.unsqueeze(omegak, dim=2)
            pRk = torch.squeeze(torch.matmul(MR, omegak_aug), dim=2)
            vk_aug = torch.unsqueeze(vk, dim=2)
            pxk = torch.squeeze(torch.matmul(Mx, vk_aug), dim=2)
            # q_p = torch.cat((qk, pk), dim=1)
            # q, p = torch.split(q_p, [self.rotmatdim, self.angveldim], dim=1)
            # M_q_inv = self.M_net(qk)
            # V_q = self.V_net(q)

            V_qk = m*self.G*qk[:,2]
            dVxRk =  torch.autograd.grad(V_qk.sum(), qk, create_graph=True)[0]
            dVxk, dVRk = torch.split(dVxRk, [self.xdim, self.Rdim], dim=1)
            dVRk = dVRk.view(-1, 3, 3)
            SMk = torch.matmul(torch.transpose(dVRk, 1, 2), Rk) - torch.matmul(torch.transpose(Rk, 1, 2), dVRk)
            Mk = torch.stack((SMk[:, 2, 1], SMk[:, 0, 2], SMk[:, 1, 0]),dim=1)


            alpha = 0.5
            a = self.h*pRk + (1-alpha)*self.h**2 * Mk + self.h *torch.squeeze(fRk_minus)
            #####################################################################################
            qk_next_data, _, _ = torch.split(x_next_data, [self.posedim, self.twistdim, self.udim], dim=1)

            _, qRk_next_data = torch.split(qk_next_data, [self.xdim, self.Rdim], dim=1)
            Rk_next_data = qRk_next_data.view(-1, 3, 3)
            Fk = torch.matmul(torch.transpose(Rk, 1, 2), Rk_next_data)

            Sv = torch.matmul(Fk, Jd) - torch.matmul(Jd, torch.transpose(Fk, 1, 2))
            v = vee_map_batch(Sv)

            implicit_eq_loss = L2_loss(a, v)
            #####################################################################################
            Rk_next = torch.matmul(Rk, Fk)
            qRk_next = Rk_next.view(-1, 9)
            qxk_next = qxk + self.h*torch.squeeze(torch.matmul(Mx_inv, torch.unsqueeze(pxk, dim=2))) - \
                       self.h*torch.squeeze(torch.matmul(Mx_inv, torch.matmul(Rk, fxk_minus)))  - \
                       ((1-alpha)*(self.h**2))*torch.squeeze(torch.matmul(Mx_inv,torch.unsqueeze(dVxk,dim=2)))
            qk_next = torch.cat((qxk_next, qRk_next), dim = 1)

            V_qk_next = m*self.G*qk_next[:,2]
            dVqk_next =  torch.autograd.grad(V_qk_next.sum(), qk_next, create_graph=True)[0]
            dVxk_next, dVRk_next = torch.split(dVqk_next, [self.xdim, self.Rdim], dim=1)
            dVRk_next = dVRk_next.view(-1, 3, 3)
            SMk_next = torch.matmul(torch.transpose(dVRk_next, 1, 2), Rk_next) - \
                       torch.matmul(torch.transpose(Rk_next, 1, 2), dVRk_next)
            Mk_next = torch.stack((SMk_next[:, 2, 1], SMk_next[:, 0, 2], SMk_next[:, 1, 0]), dim = 1)

            FkT = torch.transpose(Fk, 1, 2)
            pRk_next = torch.matmul(FkT, pRk[:,:,None]) + (1-alpha)*self.h*torch.matmul(FkT, Mk[:,:,None]) +\
                       alpha*self.h*Mk_next[:,:,None] + torch.matmul(FkT, fRk_minus) + fRk_plus
            pxk_next = pxk - (1-alpha)*self.h*dVxk - alpha*self.h*dVxk_next + \
                       torch.squeeze(torch.matmul(Rk, fxk_minus)) + torch.squeeze(torch.matmul(Rk_next, fxk_plus))
            omegak_next = torch.matmul(MR_inv, pRk_next)
            omegak_next = omegak_next[:,:,0]
            vk_next = torch.matmul(Mx_inv, torch.unsqueeze(pxk_next, dim = 2))
            vk_next = vk_next[:,:,0]

            return torch.cat((qk_next, vk_next, omegak_next, uk), dim=1), implicit_eq_loss


    def predict_traininga(self, step_num, x, gt = False):
        xseq = x[None,:,:]
        curx = x
        for i in range(step_num):
            if gt:
                nextx = self.forward_gt(curx)
            else:
                nextx = self.forward_traininga(curx)
            curx = nextx
            xseq = torch.cat((xseq, curx[None,:,:]), dim = 0)

        return xseq

    def predict_trainingb(self, step_num, traj, gt = False):
        xseq = traj[0,:,:]
        xseq = xseq[None, :, :]
        curx = traj[0,:,:]
        total_implicit_loss = 0
        for i in range(step_num):
            nextx_data = traj[i+1,:,:]
            if gt:
                nextx, implicit_loss = self.forward_gt_trainingb(curx, nextx_data)
                total_implicit_loss += implicit_loss
            else:
                nextx, implicit_loss = self.forward_trainingb(curx, nextx_data)
                total_implicit_loss += implicit_loss
            curx = nextx
            curx[:,18:22] = traj[i+1, :,18:22]
            xseq = torch.cat((xseq, curx[None,:,:]), dim = 0)

        return xseq, total_implicit_loss
