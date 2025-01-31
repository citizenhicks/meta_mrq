import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Networks (Based on Appendix B.2)
class StateEncoder(nn.Module):
    def __init__(self, state_dim, zs_dim, enc_hdim):
        super().__init__()
        self.zs_dim = zs_dim
        self.activ = nn.ELU()

        self.state_mlp1 = nn.Linear(state_dim, enc_hdim)
        self.state_mlp2 = nn.Linear(enc_hdim, enc_hdim)
        self.state_mlp3 = nn.Linear(enc_hdim, zs_dim)

    def forward(self, state):
        zs = self.activ(self.state_mlp1(state))
        zs = self.activ(self.state_mlp2(zs))
        zs = self.activ(self.state_mlp3(zs))
        return zs


class StateActionEncoder(nn.Module):
    def __init__(self, action_dim, za_dim, zs_dim, zsa_dim, enc_hdim, output_dim):
        super().__init__()
        self.za_dim = za_dim
        self.zsa_dim = zsa_dim
        self.zs_dim = zs_dim
        self.activ = nn.ELU()

        self.za_layer = nn.Linear(action_dim, za_dim)  # za encoder
        self.zsa1 = nn.Linear(zs_dim + za_dim, enc_hdim)
        self.zsa2 = nn.Linear(enc_hdim, enc_hdim)
        self.zsa3 = nn.Linear(enc_hdim, zsa_dim)
        self.model = nn.Linear(zsa_dim, output_dim)

    def forward(self, zs, action):
        za = self.activ(self.za_layer(action))
        zsa = torch.cat([zs, za], dim=1)
        zsa = self.activ(self.zsa1(zsa))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return self.model(zsa), zsa


class ValueNetwork(nn.Module):  # Value Network
    def __init__(self, zsa_dim, value_hdim):
        super().__init__()
        self.value_activ = nn.ELU()

        self.mlp = nn.Linear(zsa_dim, value_hdim)
        self.mlp2 = nn.Linear(value_hdim, value_hdim)
        self.mlp3 = nn.Linear(value_hdim, value_hdim)
        self.mlp4 = nn.Linear(value_hdim, 1)

    def forward(self, zsa):
        q = self.value_activ(self.mlp(zsa))
        q = self.value_activ(self.mlp2(q))
        q = self.value_activ(self.mlp3(q))
        return self.mlp4(q)

class Value(nn.Module):
    def __init__(self, zsa_dim, value_hdim=512):
        super().__init__()
        self.q1 = ValueNetwork(zsa_dim, value_hdim)
        self.q2 = ValueNetwork(zsa_dim, value_hdim)

    def forward(self, zsa):
        # Each one returns [batch_size, 1], -> [batch_size, 2]
        return torch.cat([self.q1(zsa), self.q2(zsa)], dim=1)


class PolicyNetwork(nn.Module):  # Policy Network
    def __init__(self, zs_dim, policy_hdim, action_dim, discrete):
        super().__init__()
        self.policy_activ = nn.ReLU()

        self.mlp = nn.Linear(zs_dim, policy_hdim)
        self.mlp2 = nn.Linear(policy_hdim, policy_hdim)
        self.mlp3 = nn.Linear(policy_hdim, action_dim)

        self.discrete = discrete
        if discrete:
            self.final_activ = lambda x: F.gumbel_softmax(x, tau=10, hard=True)
        else:
            self.final_activ = torch.tanh

    def forward(self, zs):
        a = self.policy_activ(self.mlp(zs))
        a = self.policy_activ(self.mlp2(a))
        logits = self.mlp3(a)
        return self.final_activ(logits), a
