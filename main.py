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
from lib.options import config
from lib.utils import evaluate
from lib.visualization import visualize_TSNE


def discrepancy(out1, out2):
    return torch.mean(torch.abs(out1 - out2))

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

    base_net, pred_net_1, pred_net_2 = pose_resnet.get_pose_net(config)
    base_net = base_net.to(device)
    pred_net_1 = pred_net_1.to(device)
    pred_net_2 = pred_net_2.to(device)

    optim_b = torch.optim.Adam(base_net.parameters(), config.TRAIN.BASE_LR,
                               weight_decay=config.TRAIN.WEIGHT_DECAY)
    optim_p1 = torch.optim.Adam(pred_net_1.parameters(), config.TRAIN.BASE_LR,
                                weight_decay=config.TRAIN.WEIGHT_DECAY)
    optim_p2 = torch.optim.Adam(pred_net_2.parameters(), config.TRAIN.BASE_LR,
                                weight_decay=config.TRAIN.WEIGHT_DECAY)

    input_shape = (config.TRAIN.BATCH_SIZE, 3, config.MODEL.IMG_SIZE, config.MODEL.IMG_SIZE)
    logger.add_graph(base_net, input_shape, device)

    if len(config.MODEL.RESUME) > 0:
        print("=> loading checkpoint '{}'".format(config.MODEL.RESUME))
        resume_ckpt = torch.load(config.MODEL.RESUME)
        base_net.load_state_dict(resume_ckpt['base_net'])
        pred_net_1.load_state_dict(resume_ckpt['pred_net_1'])
        pred_net_2.load_state_dict(resume_ckpt['pred_net_2'])
        optim_b.load_state_dict(resume_ckpt['optim_b'])
        optim_p1.load_state_dict(resume_ckpt['optim_p1'])
        optim_p2.load_state_dict(resume_ckpt['optim_p2'])
        config.TRAIN.START_ITERS = resume_ckpt['iter']
        logger.global_step = resume_ckpt['iter']
        logger.best_metric_val = resume_ckpt['best_metric_val']

    base_net = torch.nn.DataParallel(base_net)
    pred_net_1 = torch.nn.DataParallel(pred_net_1)
    pred_net_2 = torch.nn.DataParallel(pred_net_2)

    if config.EVALUATE:
        pck05, pck2 = evaluate(base_net, target_val_loader, pred_net_1=pred_net_1, img_size=config.MODEL.IMG_SIZE,
                               vis=True, logger=logger, status='PRED_NET_1', disp_interval=config.MISC.DISP_INTERVAL)
        print("=> validate pred_net_1 pck@0.05 = {}, pck@0.2 = {}".format(pck05 * 100, pck2 * 100))

        pck05, pck2 = evaluate(base_net, target_val_loader, pred_net_2=pred_net_2, img_size=config.MODEL.IMG_SIZE,
                               vis=True, logger=logger, status='PRED_NET_2', disp_interval=config.MISC.DISP_INTERVAL)
        print("=> validate pred_net_2 pck@0.05 = {}, pck@0.2 = {}".format(pck05 * 100, pck2 * 100))

        pck05, pck2 = evaluate(base_net, target_val_loader, pred_net_1=pred_net_1, pred_net_2=pred_net_2,
                               img_size=config.MODEL.IMG_SIZE, vis=True, logger=logger, status='AVG',
                               disp_interval=config.MISC.DISP_INTERVAL)
        print("=> validate ensemble(avg) pck@0.05 = {}, pck@0.2 = {}".format(pck05 * 100, pck2 * 100))

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

            #################################
            #            STAGE 1            #
            #################################
            # forward source
            feat_s = base_net(source_inputs)
            output_s1 = pred_net_1(feat_s)
            output_s2 = pred_net_2(feat_s)

            loss_s1 = criterion(output_s1, source_heats).sum() / source_inputs.size(0)
            loss_s2 = criterion(output_s2, source_heats).sum() / source_inputs.size(0)

            loss_s = loss_s1 + loss_s2

            # update F, C1, C2
            optim_b.zero_grad(); optim_p1.zero_grad(); optim_p2.zero_grad()
            loss_s.backward()
            optim_b.step(); optim_p1.step(); optim_p2.step()

            #################################
            #            STAGE 2            #
            #################################
            # forward source again
            feat_s = base_net(source_inputs)
            output_s1 = pred_net_1(feat_s)
            output_s2 = pred_net_2(feat_s)

            loss_s1 = criterion(output_s1, source_heats).sum() / source_inputs.size(0)
            loss_s2 = criterion(output_s2, source_heats).sum() / source_inputs.size(0)

            loss_s = loss_s1 + loss_s2

            # forward target
            feat_t = base_net(target_inputs)
            output_t1 = pred_net_1(feat_t)
            output_t2 = pred_net_2(feat_t)
            loss_dis = discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis

            # update C1, C2
            optim_b.zero_grad(); optim_p1.zero_grad(); optim_p2.zero_grad()
            loss.backward()
            optim_p1.step(); optim_p2.step()

            #################################
            #            STAGE 3            #
            #################################
            for i in range(config.TRAIN.MCD.NUM_K):
                # forward target
                feat_t = base_net(target_inputs)
                output_t1 = pred_net_1(feat_t)
                output_t2 = pred_net_2(feat_t)
                loss_dis = discrepancy(output_t1, output_t2)

                # update F
                optim_b.zero_grad()
                loss_dis.backward()
                optim_b.step()

            # val
            if logger.global_step % config.MISC.TEST_INTERVAL == 0:
                # visualize TSNE for layer4 feature embedding
                visualize_TSNE(torch.cat([feat_s, feat_t], dim=0).detach(), logger)

                val_loader = [{'src': source_val_loader},
                              {'tgt': target_val_loader}]
                pred_nets = [{'pred_net_1': pred_net_1, 'pred_net_2': None, 'status': 'PRED_NET_1'},
                             {'pred_net_1': None, 'pred_net_2': pred_net_2, 'status': 'PRED_NET_2'},
                             {'pred_net_1': pred_net_1, 'pred_net_2': pred_net_2, 'status': 'AVG'}]
                for val_dict in val_loader:
                    for net_dict in pred_nets:
                        domain, loader = list(val_dict.items())[0]
                        pck05, pck2 = evaluate(base_net, loader, img_size=config.MODEL.IMG_SIZE, vis=True,
                                               logger=logger, disp_interval=config.MISC.DISP_INTERVAL,
                                               show_gt=(logger.global_step == 0), is_target=(domain == 'tgt'),
                                               **net_dict)
                        logger.add_scalar('{}_{}_pck@0.05'.format(domain, net_dict['status']), pck05 * 100)
                        logger.add_scalar('{}_{}_pck@0.2'.format(domain, net_dict['status']), pck2 * 100)
                        if domain == 'tgt':
                            logger.save_ckpt(state={
                                'base_net': base_net.module.state_dict(),
                                'pred_net_1': pred_net_1.module.state_dict(),
                                'pred_net_2': pred_net_2.module.state_dict(),
                                'optim_b': optim_b.state_dict(),
                                'optim_p1': optim_p1.state_dict(),
                                'optim_p2': optim_p2.state_dict(),
                                'iter': logger.global_step,
                                'cur_status': net_dict['status'],
                                'best_metric_val': logger.best_metric_val,
                            }, cur_metric_val=pck05)

            logger.step(1)
            total_progress_bar.update(1)

            # log
            logger.add_scalar('loss_s', loss_s.item())
            logger.add_scalar('loss_tgt_dis', loss_dis.item())

        epoch += 1

    total_progress_bar.close()


if __name__ == '__main__':
    main()
    print("=> exit normally")
