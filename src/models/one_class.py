import torch
import numpy as np

class BALADLoss(torch.nn.Module):
    """

    Parameters
    ----------
    c: torch.Tensor
        Center of the pre-defined hyper-sphere in the representation space

    reduction: str, optional (default='mean')
        choice = [``'none'`` | ``'mean'`` | ``'sum'``]
            - If ``'none'``: no reduction will be applied;
            - If ``'mean'``: the sum of the output will be divided by the number of
            elements in the output;
            - If ``'sum'``: the output will be summed

    """

    def __init__(self, c, eta=1.0, eps=1e-6, R=0, nu=0.1, reduction='mean'):
        super(BALADLoss, self).__init__()
        self.c = c
        self.reduction = reduction
        self.eta = eta
        self.eps = eps
        self.R = torch.tensor(R, device=self.c.device)
        self.nu = nu

    def forward(self, rep, semi_targets=None, up_R=False, reduction=None):
        dist = torch.sum((rep - self.c) ** 2, dim=1)
        scores = dist - self.R ** 2
        b_loss = self.R ** 2 + (1 / self.nu) * torch.max(torch.zeros_like(scores), scores)
        if semi_targets is not None:
            loss = torch.where(semi_targets != 1, dist,
                               self.eta * ((dist + self.eps) ** -1.))
        else:
            loss = b_loss

        if up_R:
            self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.c.device)

        if reduction is None:
            reduction = self.reduction
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss

def get_radius(dist: torch.Tensor, nu: float):
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)