# -*-coding:UTF-8-*-

import os
import random

import torch.backends.cudnn as cudnn
import tqdm

from lib import Mytransforms
from lib.dataset import HandKptDataset
from lib.logger import Logger
from lib.model import pose_resnet
from lib.model.jan import JAN
from lib.model.adv import *
from lib.options import config
from lib.utils import evaluate


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

    # train
    source_loader = torch.utils.data.DataLoader(
        source_dset,
        batch_size=config.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=config.MISC.WORKERS, pin_memory=True)
    target_loader = torch.utils.data.DataLoader(
        target_dset,
        batch_size=config.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=config.MISC.WORKERS, pin_memory=True)


    # val
    source_val_loader = torch.utils.data.DataLoader(
        source_val_dset,
        batch_size=config.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=config.MISC.WORKERS, pin_memory=True)
    target_val_loader = torch.utils.data.DataLoader(
        target_val_dset,
        batch_size=config.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=config.MISC.WORKERS, pin_memory=True)

    logger = Logger(ckpt_path=os.path.join(config.DATA.CKPT_PATH, config.PROJ_NAME),
                    tsbd_path=os.path.join(config.DATA.VIZ_PATH, config.PROJ_NAME))

    net = pose_resnet.get_pose_net(config).to(device)
    optimizer = torch.optim.Adam(net.parameters(), config.TRAIN.BASE_LR,
                                 weight_decay=config.TRAIN.WEIGHT_DECAY)

    input_shape = (config.TRAIN.BATCH_SIZE, 3, config.MODEL.IMG_SIZE, config.MODEL.IMG_SIZE)
    logger.add_graph(net, input_shape, device)

    if len(config.MODEL.RESUME) > 0:
        print("=> loading checkpoint '{}'".format(config.MODEL.RESUME))
        resume_ckpt = torch.load(config.MODEL.RESUME)
        net.load_state_dict(resume_ckpt['net'])
        optimizer.load_state_dict(resume_ckpt['optim'])
        config.TRAIN.START_ITERS = resume_ckpt['iter']
        logger.global_step = resume_ckpt['iter']
        logger.best_metric_val = resume_ckpt['best_metric_val']
    net = torch.nn.DataParallel(net)

    if config.EVALUATE:
        pck05, pck2 = evaluate(net, target_val_loader, img_size=config.MODEL.IMG_SIZE, vis=True,
                       logger=logger, disp_interval=config.MISC.DISP_INTERVAL)
        print("=> validate pck@0.05 = {}, pck@0.2 = {}".format(pck05 * 100, pck2 * 100))
        return

    criterion = nn.SmoothL1Loss(reduction='none').to(device)

    total_progress_bar = tqdm.tqdm(desc='Train iter', ncols=80,
                                   total=config.TRAIN.MAX_ITER,
                                   initial=config.TRAIN.START_ITERS)
    epoch = 0

    while logger.global_step < config.TRAIN.MAX_ITER:
        for (source_data, target_data) in tqdm.tqdm(
                zip(source_loader, target_loader),
                total=min(len(source_loader), len(target_loader)),
                desc='Current epoch', ncols=80, leave=False):

            source_inputs, source_heats, _ = source_data
            target_inputs, *_ = target_data

            # data preparation
            source_inputs = source_inputs.to(device)
            source_heats = source_heats.to(device)

            target_inputs = target_inputs.to(device)

            source_size = source_inputs.size(0)
            union_inputs = torch.cat([source_inputs, target_inputs], dim=0)

            # forward
            union_preds = net(union_inputs)

            # calc loss
            regress_loss = criterion(union_preds[:source_size], source_heats).sum() / source_inputs.size(0)
            loss = regress_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # val
            if logger.global_step % config.MISC.TEST_INTERVAL == 0:
                pck05, pck2 = evaluate(net, source_val_loader, img_size=config.MODEL.IMG_SIZE, vis=True,
                                       logger=logger, disp_interval=config.MISC.DISP_INTERVAL,
                                       show_gt=(logger.global_step == 0), is_target=False)
                logger.add_scalar('src_pck@0.05', pck05 * 100)
                logger.add_scalar('src_pck@0.2', pck2 * 100)

                pck05, pck2 = evaluate(net, target_val_loader, img_size=config.MODEL.IMG_SIZE, vis=True,
                                       logger=logger, disp_interval=config.MISC.DISP_INTERVAL,
                                       show_gt=(logger.global_step == 0), is_target=True)
                logger.add_scalar('tgt_pck@0.05', pck05 * 100)
                logger.add_scalar('tgt_pck@0.2', pck2 * 100)

                logger.save_ckpt(state={
                    'net': net.module.state_dict(),
                    'optim': optimizer.state_dict(),
                    'iter': logger.global_step,
                    'best_metric_val': logger.best_metric_val,
                }, cur_metric_val=pck05)

            logger.step(1)
            total_progress_bar.update(1)

            # log
            logger.add_scalar('regress_loss', regress_loss.item())
            logger.add_scalar('loss', loss.item())

        epoch += 1

    total_progress_bar.close()


if __name__ == '__main__':
    main()
    print("=> exit normally")
