import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layer.sync_batchnorm import SynchronizedBatchNorm2d


# class GradientReverseLayer(torch.autograd.Function):
#     def __init__(self, iter_num=0, alpha=10.0, low_value=0.0, high_value=1.0, max_iter=2000.0):
#         self.iter_num = iter_num
#         self.alpha = alpha
#         self.low_value = low_value
#         self.high_value = high_value
#         self.max_iter = max_iter
#
#     def forward(self, inputs):
#         self.iter_num += 1
#         output = inputs * 1.0
#         return output
#
#     def backward(self, grad_output):
#         self.coeff = np.float(
#             2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
#                         self.high_value - self.low_value) + self.low_value) * 1000.
#         print('grl coeff: {}'.format(self.coeff))
#         return -self.coeff * grad_output

class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=10.0, low_value=0.0, high_value=1.0, max_iter=2000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                    self.high_value - self.low_value) + self.low_value) * 1000.
        return -self.coeff * grad_output


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=SynchronizedBatchNorm2d, use_sigmoid=True):
        super(PixelDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, inputs):
        return self.net(inputs)


class DCDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, use_sigmoid=True):
        super(DCDiscriminator, self).__init__()

        # self.grl_layer = GradientReverseLayer()
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        # x = self.grl_layer(x)
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.classifier(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x


class MultiPixelDiscriminator(nn.Module):
    def __init__(self, input_nc, nstage=6, ndf=64, norm_layer=SynchronizedBatchNorm2d, use_sigmoid=True):
        super(MultiPixelDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        self.nstage = nstage

        self.conv1_layers = nn.ModuleList()
        self.conv2_layers = nn.ModuleList()
        self.clsfy_layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.use_sigmoid = use_sigmoid

        # self.conv1_layers += [nn.Sequential(nn.Conv2d(512, ndf, kernel_size=3, stride=1, padding=0),
        #                                     self.leaky_relu)]
        # TODO: use kernel_size=1 could work
        for _ in range(self.nstage):
            self.conv1_layers.append(nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=0),
                                                   self.leaky_relu))
            self.conv2_layers.append(
                nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=0, bias=use_bias),
                              norm_layer(ndf * 2),
                              self.leaky_relu))
            self.clsfy_layers.append(nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias))

    def forward(self, feats):
        assert len(feats) == self.nstage, "Do not support {} inputs features!".format(len(feats))
        output = []
        for i, feat in enumerate(feats):
            x = self.conv1_layers[i](feat)
            x = self.conv2_layers[i](x)
            x = self.clsfy_layers[i](x)
            if self.use_sigmoid:
                output.append(F.sigmoid(x))
            else:
                output.append(x)
        outputs = torch.cat(output, 1)
        return outputs


class MultiDCDiscriminator(nn.Module):
    def __init__(self, input_nc, nstage=6, ndf=64, multi_task=False, use_sigmoid=False, is_cpm=False, special_dim=None):
        assert not is_cpm or (is_cpm and special_dim is not None), 'if use labels, `special_dim` should be specified'

        super(MultiDCDiscriminator, self).__init__()
        self.nstage = nstage
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.grl_layer = GradientReverseLayer()

        if multi_task:
            self.conv1_layers = nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                                              self.leaky_relu)
            self.conv2_layers = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                              self.leaky_relu)
            self.conv3_layers = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                              self.leaky_relu)
            self.conv4_layers = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                              self.leaky_relu)
        else:
            self.conv1_layers = nn.ModuleList()
            self.conv2_layers = nn.ModuleList()
            self.conv3_layers = nn.ModuleList()
            self.conv4_layers = nn.ModuleList()
        self.clsfy_layers = nn.ModuleList()

        # self.conv1_layers.append(nn.Sequential(nn.Conv2d(512, ndf, kernel_size=4, stride=2, padding=1),
        #                                     self.leaky_relu))

        for i in range(self.nstage):
            if not multi_task:
                if is_cpm and i == 0:
                    self.conv1_layers.append(
                        nn.Sequential(nn.Conv2d(special_dim, ndf, kernel_size=4, stride=2, padding=1),
                                      self.leaky_relu))
                else:
                    self.conv1_layers.append(nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                                                           self.leaky_relu))
                self.conv2_layers.append(nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                                       self.leaky_relu))
                self.conv3_layers.append(nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                                       self.leaky_relu))
                self.conv4_layers.append(nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                                       self.leaky_relu))
            self.clsfy_layers.append(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))

        self.use_sigmoid = use_sigmoid
        self.multi_task = multi_task

        # he initialize
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.2)

    def forward(self, feats, iter_num):
        assert len(feats) == self.nstage, "Do not support {} inputs features!".format(len(feats))
        output = []
        for i, feat in enumerate(feats):
            # feat = GradientReverseLayer(iter_num=iter_num)(feat)
            if not self.multi_task:
                x = self.conv1_layers[i](feat)
                x = self.conv2_layers[i](x)
                x = self.conv3_layers[i](x)
                x = self.conv4_layers[i](x)
            else:
                x = self.conv1_layers(feat)
                x = self.conv2_layers(x)
                x = self.conv3_layers(x)
                x = self.conv4_layers(x)
            x = self.clsfy_layers[i](x)
            if self.use_sigmoid:
                output.append(F.sigmoid(x))
            else:
                output.append(x)
        outputs = torch.cat(output, 1)
        return outputs


class WeightedConcat(nn.Module):
    def __init__(self, coeff=0.):
        super(WeightedConcat, self).__init__()
        self.coeff = nn.Parameter(torch.tensor([coeff], dtype=torch.float))

    def forward(self, input_x, input_y):
        """
            remember to pass (feats, heats) to this module when trade off between them
            remember to pass (stage1, stage2) when trade off between stages
        """
        w = F.sigmoid(self.coeff)
        if isinstance(input_x, list) and isinstance(input_y, list):
            combined = [torch.cat([y * (1 - w), x * w], dim=1) \
                        for y, x in zip(input_y, input_x)]
        elif torch.is_tensor(input_x) and torch.is_tensor(input_y):
            combined = torch.cat([input_y * (1 - w), input_x * w], dim=1)
        else:
            raise ValueError('Expect two `list` or two `tensor`, got {} and {}'.format(type(input_x), type(input_y)))
        return combined, w


class RMMLayer(torch.autograd.Function):
    def __init__(self, input_dim_list=list(), output_dim=512):
        self.input_num = len(input_dim_list)
        assert self.input_num == 2, 'Only support two inputs'

        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]
        for val in self.random_matrix:
            val.requires_grad = False

    def forward(self, input_list):
        assert len(input_list) == self.input_num, 'Only support two inputs'
        for input_ in input_list:
            assert input_.ndimension() == 4, 'Tensors must be in 4 dims, which is [N C H W]'

        N = input_list[0].size(0)
        H = input_list[0].size(2)
        W = input_list[0].size(3)
        channel_list = [input_.size(1) for input_ in input_list]

        # [N C H W] -> [N*H*W C]
        input_list_2d = [input_.permute(0, 2, 3, 1).contiguous().view(-1, channel_list[i]) \
                         for i, input_ in enumerate(input_list)]
        output_list = [torch.mm(input_list_2d[i], self.random_matrix[i]) for i in range(self.input_num)]

        # hadamard production
        result = output_list[0] * output_list[1] / float(self.output_dim) ** 0.5

        return result.view(N, H, W, self.output_dim).permute(0, 3, 1, 2)

    def to(self, device):
        self.random_matrix = [val.to(device) for val in self.random_matrix]
