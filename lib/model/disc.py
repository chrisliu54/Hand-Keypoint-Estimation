import torch


def smooth_l1_loss(input, target, beta=1, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def discrepancy(source_data, target_data, h1, h2, criterion=smooth_l1_loss):
    src_h1 = h1(source_data)
    src_h2 = h2(source_data)

    tgt_h1 = h1(target_data)
    tgt_h2 = h2(target_data)

    src_disc = criterion(src_h1, src_h2)
    tgt_disc = criterion(tgt_h1, tgt_h2)
    return torch.abs(src_disc - tgt_disc)
