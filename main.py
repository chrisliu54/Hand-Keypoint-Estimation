# -*-coding:UTF-8-*-

import os
import random

import torch.backends.cudnn as cudnn
import tqdm

from lib import Mytransforms
from lib.dataset import HandKptDataset
from lib.logger import Logger
from lib.model import pose_resnet
from lib.model.adv import *
from lib.options import config, args
from lib.utils import evaluate, adjust_learning_rate


def main():
    cudnn.benchmark = True
    device = torch.device('cpu' if args.gpu is None or not torch.cuda.is_available() \
                              else 'cuda:{}'.format(args.gpu[0]))

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
    source_dset = HandKptDataset(config.DATA.SOURCE_TRAIN_DIR, config.DATA.SOURCE_TRAIN_LBL_FILE, 8,  # TODO: stride
                                 transformer=train_transformer)

    # target_dset = HandKptDataset(config.DATA.TARGET_TRAIN_DIR, config.DATA.TARGET_TRAIN_LBL_FILE, 8,
    #                              transformer=train_transformer)

    val_dset = HandKptDataset(config.DATA.VAL_DIR, config.DATA.VAL_LBL_FILE, 8,
                              transformer=test_transformer)

    # source only
    train_loader = torch.utils.data.DataLoader(
        source_dset,
        batch_size=config.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=config.MISC.WORKERS, pin_memory=True)

    # val
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=config.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=config.MISC.WORKERS, pin_memory=True)

    logger = Logger(ckpt_path=os.path.join(config.DATA.CKPT_PATH, config.PROJ_NAME),
                    tsbd_path=os.path.join(config.DATA.VIZ_PATH, config.PROJ_NAME))

    net = pose_resnet.get_pose_net(config).to(device)
    optimizer = torch.optim.Adam(net.parameters(), config.TRAIN.BASE_LR,
                                 weight_decay=config.TRAIN.WEIGHT_DECAY)

    input_shape = (config.TRAIN.BATCH_SIZE, 3, config.MODEL.IMG_SIZE, config.MODEL.IMG_SIZE)
    logger.add_graph(net, input_shape, device)

    if args.resume is not None:
        print("=> loading checkpoint '{}'".format(args.resume))
        resume_ckpt = torch.load(args.resume)
        net.load_state_dict(resume_ckpt['net'])
        optimizer.load_state_dict(resume_ckpt['optim'])
        config.TRAIN.START_ITERS = resume_ckpt['iter']
        logger.best_metric_val = resume_ckpt['best_metric_val']
    net = torch.nn.DataParallel(net, device_ids=args.gpu)

    criterion = nn.SmoothL1Loss(size_average=False, reduce=False).to(device)

    iters = config.TRAIN.START_ITERS
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=config.TRAIN.MAX_ITER, initial=config.TRAIN.START_ITERS)
    epoch = 0

    while iters < config.TRAIN.MAX_ITER:
        for (stu_inputs, stu_heatmap, _) in tqdm.tqdm(
                train_loader, total=len(train_loader),
                desc='Current epoch', ncols=80, leave=False):

            logger.step(1)

            # adjust learning rate
            learning_rate = adjust_learning_rate(optimizer, iters, config)

            stu_inputs = stu_inputs.to(device)
            stu_heatmap = stu_heatmap.to(device)

            stu_heats = net(stu_inputs)

            loss = criterion(stu_heats, stu_heatmap).sum() / stu_inputs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # val
            if iters % config.MISC.TEST_INTERVAL == 0:
                # TODO: logging validation `loss` and training pck if necessary
                pck = evaluate(net, val_loader, img_size=config.MODEL.IMG_SIZE, vis=True, logger=logger,
                               disp_interval=config.MISC.DISP_INTERVAL)
                logger.add_scalar('pck@0.2', pck * 100)

                logger.save_ckpt(state={
                    'net': net.module.state_dict(),
                    'optim': optimizer.state_dict(),
                    'iter': iters,
                    'best_metric_val': logger.best_metric_val,
                    'global_step': logger.global_step
                }, cur_metric_val=pck)

            iters += 1
            total_progress_bar.update(1)

            # log
            logger.add_scalar('regress_loss', loss.item())
            logger.add_scalar('lr', learning_rate)

        epoch += 1


if __name__ == '__main__':
    main()
