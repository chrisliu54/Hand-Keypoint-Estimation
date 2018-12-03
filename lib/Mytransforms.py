# -*-coding:UTF-8-*-
from __future__ import division

import collections
import numbers
import random

import cv2
import numpy as np
import torch

from lib.options import config


def normalize(tensor, mean, std):
    """Normalize a ``torch.tensor``

    Args:
        tensor (torch.tensor): tensor to be normalized.
        mean: (list): the mean of BGR
        std: (list): the std of BGR
    
    Returns:
        Tensor: Normalized tensor.
    """
    # TODO: does this make sense?
    # (Mytransforms.to_tensor(img), [128.0, 128.0, 128.0], [256.0, 256.0, 256.0]) mean, std

    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.

    h , w , c -> c, h, w

    See ``ToTensor`` for more details.

    Args:
        pic (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    img = torch.from_numpy(pic.transpose((2, 0, 1)))

    return img.float()


def resize(img, kpt, ratio, is_fixed=False):
    """Resize the ``numpy.ndarray`` and points as ratio.

    Args:
        img    (numpy.ndarray):   Image to be resized.
        kpt    (list):            Keypoints to be resized.
        ratio  (tuple or number): the ratio to resize.

    Returns:
        numpy.ndarray: Resized image.
        lists:         Resized keypoints.
    """

    if not (isinstance(ratio, numbers.Number) or (isinstance(ratio, collections.Iterable) and len(ratio) == 2)):
        raise TypeError('Got inappropriate ratio arg: {}'.format(ratio))

    w = img.shape[1]
    if w < 64:
        img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - w, cv2.BORDER_CONSTANT, value=(128, 128, 128))

    if isinstance(ratio, numbers.Number):
        num = len(kpt)
        for i in range(num):
            kpt[i][0] *= ratio
            kpt[i][1] *= ratio
        return cv2.resize(img, (0, 0), fx=ratio, fy=ratio), kpt
    else:
        num = len(kpt)
        for i in range(num):
            kpt[i][0] *= ratio[0]
            kpt[i][1] *= ratio[1]

    im = np.ascontiguousarray(cv2.resize(img, (0, 0), fx=ratio[0], fy=ratio[1]))
    img_shape = (config.MODEL.IMG_SIZE, config.MODEL.IMG_SIZE)
    if is_fixed and im.shape != img_shape:
        im = np.ascontiguousarray(cv2.resize(img, img_shape, interpolation=cv2.INTER_CUBIC))

    return im, kpt


class RandomResized(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, scale_min=0.3, scale_max=1.1):
        self.scale_min = scale_min
        self.scale_max = scale_max

    @staticmethod
    def get_params(img, scale_min, scale_max, scale):
        height, width, _ = img.shape

        ratio = random.uniform(scale_min, scale_max)
        ratio = ratio * 1.0 / scale

        return ratio

    def __call__(self, img, kpt, scale):
        """
        Args:
            img     (numpy.ndarray): Image to be resized.
            kpt     (list):          keypoints to be resized.

        Returns:
            numpy.ndarray: Randomly resize image.
            list:          Randomly resize keypoints.
        """
        ratio = self.get_params(img, self.scale_min, self.scale_max, scale)

        return resize(img, kpt, ratio)


class TestResized(object):
    """Resize the given numpy.ndarray to the size for test.

    Args:
        size: the size to resize.
    """

    def __init__(self, size):
        assert (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):

        height, width, _ = img.shape

        return output_size[0] * 1.0 / width, output_size[1] * 1.0 / height

    def __call__(self, img, kpt):
        """
        Args:
            img     (numpy.ndarray): Image to be resized.
            kpt     (list):          keypoints to be resized.

        Returns:
            numpy.ndarray: Randomly resize image.
            list:          Randomly resize keypoints.
        """
        ratio = self.get_params(img, self.size)

        return resize(img, kpt, ratio, is_fixed=True)


def rotate(img, kpt, degree):
    """Rotate the ``numpy.ndarray`` and points as degree.

    Args:
        img    (numpy.ndarray): Image to be rotated.
        kpt    (list):          Keypoints to be rotated.
        degree (number):        the degree to rotate.

    Returns:
        numpy.ndarray: Resized image.
        list:          Resized keypoints.
    """

    height, width, _ = img.shape

    img_center = (width / 2.0, height / 2.0)
    rotateMat = cv2.getRotationMatrix2D(img_center, degree, 1.0)
    cos_val = np.abs(rotateMat[0, 0])
    sin_val = np.abs(rotateMat[0, 1])
    new_width = int(height * sin_val + width * cos_val)
    new_height = int(height * cos_val + width * sin_val)
    rotateMat[0, 2] += (new_width / 2.) - img_center[0]
    rotateMat[1, 2] += (new_height / 2.) - img_center[1]

    img = cv2.warpAffine(img, rotateMat, (new_width, new_height), borderValue=(128, 128, 128))

    num = len(kpt)
    for i in range(num):
        if kpt[i][2] == 0:
            continue
        x = kpt[i][0]
        y = kpt[i][1]
        p = np.array([x, y, 1])
        p = rotateMat.dot(p)
        kpt[i][0] = p[0]
        kpt[i][1] = p[1]

    return np.ascontiguousarray(img), kpt


class RandomRotate(object):
    """Rotate the inputs numpy.ndarray and points to the given degree.

    Args:
        max_degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree):
        assert isinstance(max_degree, numbers.Number)
        self.max_degree = max_degree

    @staticmethod
    def get_params(max_degree):
        """Get parameters for ``rotate`` for a random rotate.
           rotate:40

        Returns:
            number: degree to be passed to ``rotate`` for random rotate.
        """
        degree = random.uniform(-max_degree, max_degree)

        return degree

    def __call__(self, img, kpt):
        """
        Args:
            img    (numpy.ndarray): Image to be rotated.
            kpt    (list):          Keypoints to be rotated.

        Returns:
            numpy.ndarray: Rotated image.
            list:          Rotated keypoints.
        """
        degree = self.get_params(self.max_degree)

        return rotate(img, kpt, degree)


def crop(img, kpt, offset_left, offset_up, w, h):
    num = len(kpt)
    for x in range(num):
        if kpt[x][2] == 0:
            continue
        kpt[x][0] -= offset_left
        kpt[x][1] -= offset_up

    height, width, _ = img.shape
    new_img = np.empty((h, w, 3), dtype=np.float32)
    new_img.fill(128)

    st_x = 0
    ed_x = w
    st_y = 0
    ed_y = h
    or_st_x = offset_left
    or_ed_x = offset_left + w
    or_st_y = offset_up
    or_ed_y = offset_up + h

    # the person_center is in left
    if offset_left < 0:
        st_x = -offset_left
        or_st_x = 0
    if offset_left + w > width:
        ed_x = width - offset_left
        or_ed_x = width
    # the person_center is in up
    if offset_up < 0:
        st_y = -offset_up
        or_st_y = 0
    if offset_up + h > height:
        ed_y = height - offset_up
        or_ed_y = height

    new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()

    return np.ascontiguousarray(new_img), kpt


class KeyAreaCrop(object):
    """Crop the annotated hand region given numpy.ndarray.

    Args:
        margin (int): The margin around the cropped hand.
    """

    def __init__(self, margin=20):
        self.margin = margin

    def __call__(self, img, kpt):
        left = 1000000
        right = -5
        up = -5
        bottom = 1000000

        height, width, _ = img.shape

        for p in range(len(kpt)):
            assert kpt[p][2] == 1
            if kpt[p][0] < left:
                left = kpt[p][0]
            if kpt[p][0] > right:
                right = kpt[p][0]
            if kpt[p][1] < bottom:
                bottom = kpt[p][1]
            if kpt[p][1] > up:
                up = kpt[p][1]

        left = max(left - self.margin, 0)
        right = min(right + self.margin, width - 1)
        up = min(up + self.margin, height - 1)
        bottom = max(bottom - self.margin, 0)

        return crop(img, kpt, int(left), int(bottom), int(right - left) + 1, int(up - bottom) + 1)


def hflip(img, kpt):
    height, width, _ = img.shape

    img = img[:, ::-1, :]

    num = len(kpt)
    for i in range(num):
        if kpt[i][2] == 1:
            kpt[i][0] = width - 1 - kpt[i][0]

    # swap_pair = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9]]
    #
    # for x in swap_pair:
    #     temp_point = kpt[x[0]]
    #     kpt[x[0]] = kpt[x[1]]
    #     kpt[x[1]] = temp_point

    return np.ascontiguousarray(img), kpt


class RandomHorizontalFlip(object):
    """Random horizontal flip the image.

    Args:
        prob (number): the probability to flip.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, kpt):
        """
        Args:
            img    (numpy.ndarray): Image to be flipped.
            kpt    (list):          Keypoints to be flipped.

        Returns:
            numpy.ndarray: Randomly flipped image.
            list: Randomly flipped points.
        """
        if random.random() < self.prob:
            return hflip(img, kpt)
        return img, kpt


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>      RandomResized(),
        >>>      RandomRotate(40),
        >>>      RandomHorizontalFlip(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpt, scale=None):

        for t in self.transforms:
            if isinstance(t, RandomResized):
                img, kpt = t(img, kpt, scale)
            else:
                img, kpt = t(img, kpt)

        return img, kpt
