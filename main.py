# -*-coding:UTF-8-*-

import os
import random
from functools import partial
import itertools

import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from lib import Mytransforms
from lib.dataset import HandKptDataset
from lib.logger import Logger
from lib.model import pose_resnet
from lib.options import config
from lib.utils import evaluate


def iterate_eternally(iterable):
    def infinite_iterate():
        while True:
            yield iterable

    return itertools.chain.from_iterable(infinite_iterate())


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(n_iter):
    return config.TRAIN.CONSISTENCY * sigmoid_rampup(n_iter, config.TRAIN.CONSISTENCY_RAMPUP)
    # return min(config.consistency * ramps.sigmoid_rampup(epoch, config.consistency_rampup), 1)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def main():
    cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert config.MISC.TEST_INTERVAL is not 0, 'Illegal setting: config.MISC.TEST_INTERVAL = 0!'

    # set random seed
    if config.MISC.RANDOM_SEED:
        random.seed(config.MISC.RANDOM_SEED)
        np.random.seed(config.MISC.RANDOM_SEED)
        torch.manual_seed(config.MISC.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.MISC.RANDOM_SEED)

    train_transformer = Mytransforms.Compose([
        Mytransforms.KeyAreaCrop(20),
        Mytransforms.RandomRotate(40),
        Mytransforms.TestResized(config.MODEL.IMG_SIZE),
        Mytransforms.RandomHorizontalFlip()
    ])

    test_transformer = Mytransforms.Compose([
        Mytransforms.KeyAreaCrop(20),
        Mytransforms.TestResized(config.MODEL.IMG_SIZE)
    ])

    # train
    source_dset = HandKptDataset(config.DATA.SOURCE.TRAIN.DIR, config.DATA.SOURCE.TRAIN.LBL_FILE,
                                 stride=config.MODEL.HEATMAP_STRIDE, transformer=train_transformer)

    target_dset = HandKptDataset(config.DATA.TARGET.TRAIN.DIR, config.DATA.TARGET.TRAIN.LBL_FILE,
                                 stride=config.MODEL.HEATMAP_STRIDE, transformer=train_transformer)

    source_val_dset = HandKptDataset(config.DATA.SOURCE.VAL.DIR, config.DATA.SOURCE.VAL.LBL_FILE,
                                     stride=config.MODEL.HEATMAP_STRIDE, transformer=test_transformer)
    target_val_dset = HandKptDataset(config.DATA.TARGET.VAL.DIR, config.DATA.TARGET.VAL.LBL_FILE,
                                     stride=config.MODEL.HEATMAP_STRIDE, transformer=test_transformer)

    # dataloader
    target_batch_size = config.TRAIN.BATCH_SIZE - config.TRAIN.SOURCE_BATCH_SIZE
    get_dataloader = partial(torch.utils.data.DataLoader, num_workers=config.MISC.WORKERS, pin_memory=True)
    source_train_loader = get_dataloader(source_dset, batch_size=config.TRAIN.SOURCE_BATCH_SIZE, shuffle=True, drop_last=True)
    target_train_loader = get_dataloader(target_dset, batch_size=target_batch_size, shuffle=True, drop_last=True)
    source_val_loader = get_dataloader(source_val_dset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False)
    target_val_loader = get_dataloader(target_val_dset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False)
    source_train_gen = iterate_eternally(source_train_loader)
    target_train_gen = iterate_eternally(target_train_loader)
    train_gen = zip(source_train_gen, target_train_gen)

    logger = Logger(ckpt_path=os.path.join(config.DATA.CKPT_PATH, config.PROJ_NAME),
                    tsbd_path=os.path.join(config.DATA.VIZ_PATH, config.PROJ_NAME))

    net = pose_resnet.get_pose_net(config).to(device)
    ema_net = pose_resnet.get_pose_net(config).to(device)
    for param in ema_net.parameters():
        param.detach_()

    optimizer = torch.optim.Adam(net.parameters(), config.TRAIN.BASE_LR,
                                 weight_decay=config.TRAIN.WEIGHT_DECAY)

    input_shape = (config.TRAIN.BATCH_SIZE, 3, config.MODEL.IMG_SIZE, config.MODEL.IMG_SIZE)
    logger.add_graph(net, input_shape, device)

    if len(config.MODEL.RESUME) > 0:
        print("=> loading checkpoint '{}'".format(config.MODEL.RESUME))
        resume_ckpt = torch.load(config.MODEL.RESUME)
        net.load_state_dict(resume_ckpt['net'])
        ema_net.load_state_dict(resume_ckpt['ema_net'])
        optimizer.load_state_dict(resume_ckpt['optim'])
        config.TRAIN.START_ITERS = resume_ckpt['iter']
        logger.global_step = resume_ckpt['iter']
        logger.best_metric_val = resume_ckpt['best_metric_val']
    net = torch.nn.DataParallel(net)
    ema_net = torch.nn.DataParallel(ema_net)

    if config.EVALUATE:
        pck05, pck2 = evaluate(net, target_val_loader, img_size=config.MODEL.IMG_SIZE, vis=True,
                               logger=logger, disp_interval=config.MISC.DISP_INTERVAL)
        print("=> validate pck@0.05 = {}, pck@0.2 = {}".format(pck05 * 100, pck2 * 100))
        return

    criterion = nn.SmoothL1Loss(reduction='none').to(device)
    cons_criterion = nn.MSELoss(reduction='none').to(device)

    total_progress_bar = tqdm.tqdm(desc='Train iter', ncols=80,
                                   total=config.TRAIN.MAX_ITER,
                                   initial=config.TRAIN.START_ITERS)

    while logger.global_step < config.TRAIN.MAX_ITER:
        for (source_data, target_data) in train_gen:

            net.train()
            ema_net.train()

            source_inputs, source_heatmap, _ = [x.to(device) for x in source_data]
            target_inputs, target_heatmap, _ = [x.to(device) for x in target_data]

            inputs = torch.cat([source_inputs, target_inputs])

            pred_heatmap = net(inputs)
            regression_loss = criterion(pred_heatmap[:config.TRAIN.SOURCE_BATCH_SIZE], source_heatmap).sum() / config.TRAIN.SOURCE_BATCH_SIZE

            # ema
            ema_pred_heatmap = ema_net(inputs).detach()
            cons_loss = cons_criterion(pred_heatmap, ema_pred_heatmap).sum() / config.TRAIN.BATCH_SIZE

            cons_weight = get_current_consistency_weight(logger.global_step)
            loss = regression_loss + cons_weight * cons_loss

            logger.add_scalar('cons_weight', cons_weight)
            logger.add_scalar('cons_loss', cons_loss)
            logger.add_scalar('loss', loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(net, ema_net, config.TRAIN.EMA_DECAY, logger.global_step)

            # val
            if logger.global_step % config.MISC.TEST_INTERVAL == 0:
                for model, model_name in [(net, 'student'), (ema_net, 'teacher')]:
                    for dataloader, loader_name in [(source_val_loader, 'source'), (target_val_loader, 'target')]:
                        prefix = "{}_{}".format(model_name, loader_name)
                        pck05, pck2 = evaluate(model, dataloader, img_size=config.MODEL.IMG_SIZE,
                                               logger=logger, disp_interval=config.MISC.DISP_INTERVAL,
                                               show_gt=(logger.global_step == 0), vis=True, prefix=prefix)
                        logger.add_scalar('{}_pck@0.05'.format(prefix), pck05 * 100)
                        logger.add_scalar('{}_pck@0.2'.format(prefix), pck2 * 100)

                # use teacher target pck results as main metric
                logger.save_ckpt(state={
                    'net': net.module.state_dict(),
                    'optim': optimizer.state_dict(),
                    'iter': logger.global_step,
                    'best_metric_val': logger.best_metric_val,
                }, cur_metric_val=pck05)

            logger.step(1)
            total_progress_bar.update(1)

            # log
            logger.add_scalar('regress_loss', loss.item())

    total_progress_bar.close()


if __name__ == '__main__':
    main()
    print("=> exit normally")
