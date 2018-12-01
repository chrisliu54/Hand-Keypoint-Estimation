import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
from PIL import Image
from io import BytesIO


def vis_kpt(gt_pnts=None, pred_pnts=None, img=None, save_name='kpt', logger=None):
    """Visualize key points.
    Args:
      gt_pnts: (numpy.ndarray) ground truth key points
      pred_pnts: (numpy.ndarray) predicted key points
      img: (numpy.ndarary/tensor) image to visualize, whose channel is of 'BGR' order

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_bbox.py
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_image.py
      :param save_name:
      :param logger:
    """

    assert any([gt_pnts is not None, pred_pnts is not None]),\
        'VisualizationError: at least one type of key points should be supplied!'
    if img is not None:
        assert any([isinstance(img, torch.Tensor), isinstance(img, np.ndarray)]),\
            'VisualizationError: only supports torch.Tensor or np.ndarray, but got {}'.format(type(img))

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # convert BGR to RGB
    if isinstance(img, torch.Tensor):
        if img.is_cuda:
            img = img.cpu()
        idx = [i for i in range(img.size(0)-1, -1, -1)]
        img = img.index_select(0, torch.LongTensor(idx))
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
    elif isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # Plot image
    if img is not None:
        ax.imshow(img)

    if gt_pnts is not None:
        if torch.is_tensor(gt_pnts):
            gt_pnts = gt_pnts.cpu().numpy()
        else:
            gt_pnts = np.array(gt_pnts)
    if pred_pnts is not None:
        if torch.is_tensor(pred_pnts):
            pred_pnts = pred_pnts.cpu().numpy()
        else:
            pred_pnts = np.array(pred_pnts)

    for ie, e in enumerate(edges):
        # if np.all(pts[e, 2] != 0):
        rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
        if gt_pnts is not None:
            ax.plot(gt_pnts[e, 0], gt_pnts[e, 1], color=rgb)
        if pred_pnts is not None:
            ax.plot(pred_pnts[e, 0], pred_pnts[e, 1], color=rgb, linestyle='--', marker='o')

    if logger is not None:
        buffer_ = BytesIO()
        plt.savefig(buffer_, format='png', bbox_inches='tight')
        buffer_.seek(0)

        data = np.asarray(Image.open(buffer_))
        data = np.transpose(data[:, :, :3], [2, 0, 1])
        buffer_.close()

        logger.add_image(save_name, data)

    # plt.savefig('/tmp/kpt.png', bbox_inches='tight')
    plt.close(fig)
