import contextlib
import torch
import torch.nn as nn


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=1e-6, eps=8.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 1e-6)
        :param eps: hyperparameter of VAT (default: 8.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def __call__(self, model, x, pred=None):
        if pred is None:
            with torch.no_grad():
                pred = model(x)

        # criterion
        self.criterion = self.criterion.to(x.device)

        # prepare random unit tensor following gaussian dist.
        d = torch.randn(x.shape).to(x.device)
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            pred_hat = model(x + self.xi * d)
            adv_distance = self.criterion(pred_hat, pred.detach()).sum()
            # TODO: accumulates some grads over x+xi*d
            adv_distance.backward()
            d = _l2_normalize(d.grad)

        # calc LDS
        r_adv = d * self.eps
        pred_hat = model(x + r_adv)
        lds = self.criterion(pred_hat, pred.detach()).sum()

        return lds