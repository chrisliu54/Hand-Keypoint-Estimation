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


def PCK(pred, gt, img_side_len, alpha=0.2):
    """
    Calculate the PCK measure
    :param pred: predicted key points, [N, C, 2]
    :param gt: ground truth key points, [N, C, 2]
    :param img_side_len: max(width, height)
    :param alpha: normalized coefficient
    :return: PCK of current batch, number of correctly detected key points of current batch
    """
    norm_dis = alpha * img_side_len
    dis = (pred.double() - gt) ** 2
    # [N, C]
    dis = torch.sum(dis, dim=2) ** 0.5
    nkpt = (dis < norm_dis).float().sum()
    return nkpt.item() / dis.numel(), nkpt.item()


def PCK_curve_pnts(sp, pred, gt, img_side_len):
    nkpts = [PCK(pred, gt, img_side_len, alpha=a)[1] for a in sp]
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


def evaluate(model, loader, img_size, vis=False, logger=None, disp_interval=50, show_gt=True, is_target=True):
    """
    :param img_size: width/height of img_size (width == height)
    :param vis: show kpts on images or not
    :param logger: logger for tensorboardX
    :param disp_interval: interval of display
    :param model: model to be evaluated
    :param loader: dataloader to be evaluated
    :param show_gt: show ground truth or not, disabled if vis=False
    :param is_target: is from target domain or not, disabled if vis=False
    :return: PCK@0.05, PCK@0.2
    """
    device = next(model.parameters()).device
    previous_state = model.training
    model.eval()

    thresholds = np.linspace(0, 0.2, 21)

    tot_nkpts = [0] * thresholds.shape[0]
    tot_pnt = 0
    idx = 0
    domain_prefix = 'tgt' if is_target else 'src'
    with torch.no_grad():
        for (inputs, *_, gt_kpts) in tqdm.tqdm(
                loader, desc='Eval', total=len(loader), leave=False
        ):

            img_side_len = img_size
            inputs = inputs.to(device)

            # get head_maps for one image
            heats = model(inputs)

            # get predicted key points
            kpts = get_kpts(heats, img_h=img_side_len, img_w=img_side_len)

            tot_pnt += kpts.numel() / 2

            nkpts = PCK_curve_pnts(thresholds, kpts, gt_kpts[..., :2], img_side_len)
            for i in range(len(tot_nkpts)):
                tot_nkpts[i] += nkpts[i]

            if vis and idx % disp_interval == 0:
                # take the first image of the current batch for visualization
                denorm_img = denormalize(inputs[0])
                if show_gt:
                    vis_kpt(gt_pnts=gt_kpts[0, ..., :2], img=denorm_img,
                            save_name='{}_gt_kpt/{}'.format(domain_prefix, idx // disp_interval), logger=logger)
                vis_kpt(pred_pnts=kpts[0], img=denorm_img,
                        save_name='{}_pred_kpt/{}'.format(domain_prefix, idx // disp_interval), logger=logger)
            idx += 1

    # recover the state
    model.train(previous_state)

    for i in range(len(tot_nkpts)):
        tot_nkpts[i] /= tot_pnt
    if is_target:
        # draw PCK curve
        pck_line, = plt.plot(thresholds, tot_nkpts)
        plt.ylim(0, 1.)

        logger.add_figure('tgt_PCK_curve', pck_line.figure)

    return tot_nkpts[5], tot_nkpts[-1]
