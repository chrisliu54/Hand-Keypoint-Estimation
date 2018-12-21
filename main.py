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
from lib.model.disc import consistency
from lib.utils import OptimizerManager


net_name = ['from_source', 'from_target']


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
    source_train_loader = torch.utils.data.DataLoader(
        source_dset,
        batch_size=config.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=config.MISC.WORKERS, pin_memory=True)

    target_train_loader = torch.utils.data.DataLoader(
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

    nets = [pose_resnet.get_pose_net(config).to(device) for _ in range(2)]
    optimizers = [torch.optim.Adam(nets[i].parameters(), config.TRAIN.BASE_LR,
                                   weight_decay=config.TRAIN.WEIGHT_DECAY) for i in range(2)]

    input_shape = (config.TRAIN.BATCH_SIZE, 3, config.MODEL.IMG_SIZE, config.MODEL.IMG_SIZE)
    logger.add_graph(nets[0], input_shape, device)

    # resume model directly
    if len(config.MODEL.RESUME) > 0:
        print("=> loading checkpoint '{}'".format(config.MODEL.RESUME))
        resume_ckpt = torch.load(config.MODEL.RESUME)
        for i in range(2):
            nets[i].load_state_dict(resume_ckpt['net{}'.format(i)])
            optimizers[i].load_state_dict(resume_ckpt['optim{}'.format(i)])
        config.TRAIN.START_ITERS = resume_ckpt['iter']
        logger.global_step = resume_ckpt['iter']
        logger.best_metric_val = resume_ckpt['best_metric_val']
    else:
        assert config.MODEL.RESUME_PRE_SRC is not None, 'ERROR: no pretrained source model specified.'
        assert config.MODEL.RESUME_PRE_TGT is not None, 'ERROR: no pretrained target model specified.'

        print("=> loading source pretrained model '{}'".format(config.MODEL.RESUME_PRE_SRC))
        resume_pre_src = torch.load(config.MODEL.RESUME_PRE_SRC)
        nets[0].load_state_dict(resume_pre_src['net'])
        optimizers[0].load_state_dict(resume_pre_src['optim'])

        print("=> loading target pretrained model '{}'".format(config.MODEL.RESUME_PRE_TGT))
        resume_pre_tgt = torch.load(config.MODEL.RESUME_PRE_TGT)
        nets[1].load_state_dict(resume_pre_tgt['net'])
        optimizers[1].load_state_dict(resume_pre_tgt['optim'])

    for i in range(2):
        nets[i] = torch.nn.DataParallel(nets[i])

    if config.EVALUATE:
        for i in range(2):
            pck05, pck2 = evaluate(nets[i], target_val_loader, img_size=config.MODEL.IMG_SIZE, vis=True,
                           logger=logger, disp_interval=config.MISC.DISP_INTERVAL)
            print("=> validate {} net: pck@0.05 = {}, pck@0.2 = {}".format(net_name[i], pck05 * 100, pck2 * 100))
        return

    criterion = nn.SmoothL1Loss(reduction='sum').to(device)

    total_progress_bar = tqdm.tqdm(desc='Train iter', ncols=80,
                                   total=config.TRAIN.MAX_ITER,
                                   initial=config.TRAIN.START_ITERS)
    epoch = 0

    while logger.global_step < config.TRAIN.MAX_ITER:
        for (source_data, target_data) in tqdm.tqdm(
                zip(source_train_loader, target_train_loader),
                total=min(len(source_train_loader), len(target_train_loader)),
                desc='Current epoch', ncols=80, leave=False):

            source_inputs, source_heatmap, _ = source_data
            target_inputs, target_heatmap, _ = target_data
            source_inputs = source_inputs.to(device)
            source_heatmap = source_heatmap.to(device)

            # supervised learning on source domain
            optimizers[0].zero_grad()
            loss = criterion(nets[0](source_inputs), source_heatmap) / source_inputs.size(0)
            loss.backward()
            optimizers[0].step()

            # minimize the discrepancy
            with OptimizerManager(optimizers):
                disc = consistency(target_inputs, nets[0], nets[1]) * config.TRAIN.ACCORD.TRADE_OFF
                disc.backward()

            # val
            if logger.global_step % config.MISC.TEST_INTERVAL == 0:
                for i in range(2):
                    pck05, pck2 = evaluate(nets[i], source_val_loader, img_size=config.MODEL.IMG_SIZE, vis=True,
                                           logger=logger, disp_interval=config.MISC.DISP_INTERVAL,
                                           show_gt=(logger.global_step == 0), is_target=False)
                    logger.add_scalar('{}_net_src_pck@0.05'.format(net_name[i]), pck05 * 100)
                    logger.add_scalar('{}_net_src_pck@0.2'.format(net_name[i]), pck2 * 100)

                    pck05, pck2 = evaluate(nets[i], target_val_loader, img_size=config.MODEL.IMG_SIZE, vis=True,
                                           logger=logger, disp_interval=config.MISC.DISP_INTERVAL,
                                           show_gt=(logger.global_step == 0), is_target=True)
                    logger.add_scalar('{}_net_tgt_pck@0.05'.format(net_name[i]), pck05 * 100)
                    logger.add_scalar('{}_net_tgt_pck@0.2'.format(net_name[i]), pck2 * 100)

                # save according to PCK0.05 of either model
                logger.save_ckpt(state={
                    'net0': nets[0].module.state_dict(),
                    'optim0': optimizers[0].state_dict(),
                    'net1': nets[1].module.state_dict(),
                    'optim1': optimizers[1].state_dict(),
                    'iter': logger.global_step,
                    'best_metric_val': logger.best_metric_val,
                }, cur_metric_val=pck05)

            logger.step(1)
            total_progress_bar.update(1)

            # log
            logger.add_scalar('regress_loss', loss.item())
            logger.add_scalar('discrepancy', disc.item())

        epoch += 1

    total_progress_bar.close()


if __name__ == '__main__':
    main()
    print("=> exit normally")
