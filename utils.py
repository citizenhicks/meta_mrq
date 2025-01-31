# file is based on facebook research' implementation

import torch
import torch.nn.functional as F

class TwoHot:
    def __init__(self, lower: float=-10, upper: float=10, num_bins: int=65, device: torch.device=torch.device('mps')):
        # Create bins in symmetrical exp scale or whichever you had
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        # The original code:  bins = sign(bins)*(exp(abs(bins))-1)
        self.bins = torch.sign(self.bins) * (torch.exp(self.bins.abs()) - 1)
        self.num_bins = num_bins
        self.device = device

    def transform(self, x: torch.Tensor):
        """
        Returns a two-hot encoding of x wrt the self.bins
        """
        # diff shape: [batch_size, num_bins]
        diff = x - self.bins.view(1, -1).to(x.device)
        # offset large positive or negative
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, dim=1, keepdim=True).to(x.device)

        # clamp index + 1
        ind_plus1 = torch.clamp(ind + 1, max=self.num_bins-1).to(x.device)

        lower = self.bins[ind]
        upper = self.bins[ind_plus1]

        weight = (x - lower) / (upper - lower + 1e-20)

        two_hot = torch.zeros(x.size(0), self.num_bins, device=x.device)
        rows = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        two_hot.scatter_(1, ind, (1.0 - weight))
        two_hot.scatter_(1, ind_plus1, weight)
        return two_hot

    def inverse(self, x: torch.Tensor):
        """
        Weighted average of bins after softmax
        """
        probs = F.softmax(x, dim=-1)
        return (probs * self.bins.to(x.device)).sum(dim=-1, keepdim=True)

    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """
        target is shape [batch_size, 1].
        """
        # log-softmax of pred
        pred_log_softmax = F.log_softmax(pred, dim=-1)
        target_encoding = self.transform(target)
        return -(target_encoding * pred_log_softmax).sum(dim=-1, keepdim=True)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    """
    Weighted MSE, ignoring entries that are zero in mask.
    """
    return ((x - y)**2 * mask).mean()


def multi_step_reward(reward: torch.Tensor, gammas: torch.Tensor):
    """
    reward: shape [batch_size, horizon, 1]
    gammas: shape [1, horizon, 1]
    Returns sum_{t=0..horizon-1}( reward[:,t] * gammas[:,t] ), shape [batch_size, 1]
    """
    # broadcast: [batch_size, horizon, 1] * [1, horizon, 1] -> [batch_size, horizon, 1]
    out = reward * gammas
    return out.sum(dim=1)


def realign(x: torch.Tensor, discrete: bool):
    """
    If discrete, convert x to one-hot from argmax
    else clamp to [-1,1].
    """
    if discrete:
        # x shape = [batch_size, action_dim]
        with torch.no_grad():
            max_idx = torch.argmax(x, dim=1)
        one_hot = torch.zeros_like(x)
        one_hot.scatter_(1, max_idx.unsqueeze(1), 1.0)
        return one_hot
    else:
        return torch.clamp(x, -1, 1)
