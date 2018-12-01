# -*-coding:UTF-8-*-
import os
import pickle
from math import ceil

import cv2
import numpy as np
import torch.utils.data as data

from lib import Mytransforms
from lib.options import config


def load_basic_info(root_dir, file_name, img_size):
    os.makedirs(config.DATA.CACHE_DIR, exist_ok=True)
    pk_file = os.path.join(config.DATA.CACHE_DIR,
                           '-'.join(file_name.split('/')[-2:]).replace('.txt', '_' + str(img_size) + '.pkl'))

    if os.path.exists(pk_file):
        with open(pk_file, 'rb') as pf:
            info = pickle.load(pf)
    else:
        with open(pk_file, 'wb') as pf:
            imgs, kpts, scales = [[] for _ in range(3)]
            with open(file_name, 'r') as f:
                for line in f:
                    img_name, kpt_info = line.split(' ')[0], line.split(' ')[1:]
                    img_name = os.path.join(root_dir, img_name)
                    kpt_info = [[float(kpt_info[i]), float(kpt_info[i + 1]), float(kpt_info[i + 2])] \
                                for i in range(0, len(kpt_info), 3)]

                    img = cv2.imread(img_name)

                    assert img is not None, 'Img path not found: {}'.format(img_name)

                    h, w = img.shape[0], img.shape[1]
                    cur_kpt = np.transpose(np.array(kpt_info))  # (3 * 21)
                    try:
                        scale = (cur_kpt[1][cur_kpt[1] < h].max() -
                                 cur_kpt[1][cur_kpt[1] > 0].min() + 4) / img_size
                        scales.append(scale)

                        imgs.append(img_name)
                        kpts.append(kpt_info)
                    except:
                        # neglect images whose x or y is out of range
                        print('Warning: label coordinates out of image size in ' + img_name)
                info = {
                    'imgs': imgs,
                    'kpts': kpts,
                    'scales': scales
                }
                pickle.dump(info, pf)
    return info['imgs'], info['kpts'], info['scales']


def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class HandKptDataset(data.Dataset):
    """
        Args:
            root_dir (str): the path of train_val dateset.
            label_file (str): the path of the corresponding dataset's label file
        Notice:
            you have to change code to fit your own dataset except LSP

    """

    def __init__(self, root_dir, label_file, stride, transformer=None):
        self.img_list, self.kpt_list, self.scale_list = \
            load_basic_info(root_dir, label_file, config.MODEL.IMG_SIZE)

        self.stride = stride
        self.transformer = transformer
        self.sigma = 3.

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        assert img is not None, "image@{} not exist or is corrupted: {}".format(index, img_path)
        img = np.array(img, dtype=np.float32)

        kpt = self.kpt_list[index]
        scale = self.scale_list[index]

        # expand dataset
        img, kpt = self.transformer(img, kpt, scale)
        height, width, _ = img.shape

        heatmap = np.zeros((ceil(height / self.stride), ceil(width / self.stride), len(kpt)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = gaussian_kernel(size_h=height / self.stride, size_w=width / self.stride, center_x=x,
                                       center_y=y,
                                       sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i] = heat_map

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                     [256.0, 256.0, 256.0])
        heatmap = Mytransforms.to_tensor(heatmap)
        return img, heatmap, np.array(kpt)

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    dataset = HandKptDataset('/home/dataset/Kwai/CMU_Panoptic/hand_labels_synth/all_synth',
                             '/runspace/liujintao/app/Convolutional-Pose-Machines-Pytorch/dataset/labels/synth/vis_train_labels.txt',
                             transformer=Mytransforms.Compose([Mytransforms.RandomResized(),
                                                               Mytransforms.RandomRotate(40),
                                                               Mytransforms.RandomHorizontalFlip(),
                                                               ]),
                             ratio=.5,
                             stride=8)
    print('len of dataset:{}'.format(len(dataset)))
    import torch

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
                                         num_workers=4, pin_memory=True)
    for inputs, head_maps, kpts in loader:
        print(inputs.size())
        print(head_maps.size())
        print(kpts.size())
        break
