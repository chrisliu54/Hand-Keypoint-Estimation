import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

plt.switch_backend('agg')
from lib.Mytransforms import denormalize
from lib.visualization import vis_kpt


def adjust_learning_rate(optimizer, iters, config):
    base_lr = config.TRAIN.BASE_LR
    policy = config.TRAIN.LR_POLICY
    policy_parameter = config.TRAIN.POLICY_PARAMETER
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['GAMMA'] ** (iters // policy_parameter['STEP_SIZE']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['GAMMA'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['GAMMA'] * iters) ** (-policy_parameter['POWER']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['STEPVALUE']:
            if iters >= stepvalue:
                lr *= policy_parameter['GAMMA']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['MAX_ITER']) ** policy_parameter['POWER'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['GAMMA'] * (iters - policy_parameter['STEPSIZE']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['MAX_ITER']
        for stepvalue in policy_parameter['STEPVALUE']:
            if iters >= stepvalue:
                lr *= policy_parameter['GAMMA']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['GAMMA'],
                 lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['POWER'])

    if lr >= config.TRAIN.MIN_LR:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return lr
    else:
        return optimizer.param_groups[0]['lr']  # old lr


def PCK(pred, gt, tensor_size, alpha=0.2):
    """
    Calculate the PCK measure
    :param pred: predicted key points, [N, C, 2]
    :param gt: ground truth key points, [N, C, 2]
    :param tensor_size: max(width, height)
    :param alpha: normalized coefficient
    :return: PCK of current batch, number of correctly detected key points of current batch
    """
    norm_dis = alpha * tensor_size
    dis = (pred.double() - gt) ** 2
    # [N, C]
    dis = torch.sum(dis, dim=2) ** 0.5
    nkpt = (dis < norm_dis).float().sum()
    return nkpt.item() / dis.numel(), nkpt.item()


def PCK_curve_pnts(sp, pred, gt, tensor_size):
    nkpts = [PCK(pred, gt, tensor_size, alpha=a)[1] for a in sp]
    return nkpts


def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def get_kpts(maps, img_h=368.0, img_w=368.0):
    # maps (1,15,46,46) for labels
    maps = maps.clone().cpu().data.numpy()
    all_kpts = []
    for heat_map in maps:
        kpts = []
        for m in heat_map:
            h, w = np.unravel_index(m.argmax(), m.shape)
            x = int(w * img_w / m.shape[1])
            y = int(h * img_h / m.shape[0])
            kpts.append([x, y])
        all_kpts.append(kpts)
    return torch.from_numpy(np.array(all_kpts))


def evaluate(model, loader, img_size, vis=False, logger=None, disp_interval=50, show_gt=True):
    """
    :param img_size:
    :param vis:
    :param logger:
    :param disp_interval:
    :param mode:
    :param model: model to be evaluated
    :param loader: dataloader to be evaluated
    :return: PCK
    """
    device = next(model.parameters()).device
    previous_state = model.training
    model.eval()

    tot_nkpt = 0
    tot_pnt = 0
    idx = 0
    with torch.no_grad():
        for (inputs, *_, gt_kpts) in tqdm.tqdm(
                loader, desc='Eval', total=len(loader), leave=False
        ):

            tensor_size = img_size
            inputs = inputs.to(device)

            # get head_maps for one image
            heats = model(inputs)

            # get predicted key points
            kpts = get_kpts(heats, img_h=tensor_size, img_w=tensor_size)

            # print('predicted kpts  vs  gt kpts')
            # for kpt, gt_kpt in zip(kpts[0], gt_kpts[0]):
            #     print('[{}, {}] vs [{}, {}]'
            #           .format(kpt[0], kpt[1], gt_kpt[0], gt_kpt[1]))

            pck, nkpt = PCK(kpts, gt_kpts[..., :2], tensor_size)
            # print('pck = {}, nkpt = {}, pnt = {}'.format(pck * 100, nkpt, kpts.numel()/2))
            tot_nkpt += nkpt
            tot_pnt += kpts.numel() / 2

            if vis and idx % disp_interval == 0:
                # take the first image of the current batch
                denorm_img = denormalize(inputs[0])
                if show_gt:
                    vis_kpt(gt_pnts=gt_kpts[0, ..., :2], img=denorm_img,
                            save_name='gt_kpt/{}'.format(idx // disp_interval), logger=logger)
                vis_kpt(pred_pnts=kpts[0], img=denorm_img,
                        save_name='pred_kpt/{}'.format(idx // disp_interval), logger=logger)
            idx += 1

    # recover the state
    model.train(previous_state)

    return tot_nkpt / tot_pnt
