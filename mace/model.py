import torch as th
import torch.nn as nn
from torch.autograd import Variable
from params import args


class NeuralCoord(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(NeuralCoord, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU())

        self.latent_dim = args.latent_dim  # 512
        self.FCV = nn.Linear(512+183*self.n_agent, n_agent)
        self.logvar = nn.Linear(512+183*self.n_agent, self.latent_dim)
        self.mu = nn.Linear(512+183*self.n_agent, self.latent_dim)

        # z_mu and z_logvar are the parameters for the action distribution
        # we will sample from them to get the actions
        self.z_logvar = nn.Linear(self.latent_dim, self.latent_dim*n_agent)
        self.z_mu = nn.Linear(self.latent_dim, self.latent_dim*n_agent)
        self.use_cuda = False
        self.FC = nn.ModuleList(
            [nn.Linear(self.latent_dim+(512+183*self.n_agent),
                       dim_action) for i in range(self.n_agent)])

    # obs: batch_size * 3 * 84 * 84
    # fea: batch_size * (183 * n_agent)
    def forward(self, fea, obs):
        phi_s = self.cnn(obs)
        phi_s = phi_s.view(phi_s.size(0), -1)
        phi_s = th.cat((phi_s, fea), 1)
        # V: n_agent
        V = self.FCV(phi_s)
        # l_sigma: latent_dim
        l_logvar = self.logvar(phi_s)
        l_mu = self.mu(phi_s)
        if self.use_cuda:
            eps1 = th.cuda.FloatTensor(self.latent_dim).normal_()
        else:
            eps1 = th.FloatTensor(self.latent_dim).normal_()
        eps1 = Variable(eps1)
        lam = eps1 * (l_logvar * 0.5).exp() + l_mu
        # lam: self.latent_dim
        mu = self.z_mu(lam.unsqueeze(0))
        logvar = self.z_logvar(lam.unsqueeze(0))
        mu = mu.view(-1, self.latent_dim)
        logvar = logvar.view(-1, self.latent_dim)
        if self.use_cuda:
            eps2 = th.cuda.FloatTensor(self.n_agent, self.latent_dim).normal_()
        else:
            eps2 = th.FloatTensor(self.n_agent, self.latent_dim).normal_()
        eps2 = Variable(eps2)
        # z: n_agent x latent_dim
        z = mu + (logvar * 0.5).exp() * eps2
        z = th.cat((z, phi_s.repeat(self.n_agent, 1)), 1)

        prob = []
        for a in range(self.n_agent):
            prob_ = self.FC[a](z[a].unsqueeze(0))
            prob.append(prob_)

        prob = th.cat(prob)

        return V, prob, l_mu, l_logvar, mu, logvar
