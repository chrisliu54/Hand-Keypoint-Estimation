import torch


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, matrix_op=False):
    """
    :param source: (N, C, H, W)
    :param target: (N, C, H, W)
    :param kernel_mul: kernel multiplier, will be multiply to bandwidth by kernel_mul^(kernel_id - kernel_num//2)
    :param kernel_num: number of kernel
    :param fix_sigma: bandwidth of kernel if specified, or bandwidth will be the mean of L2 distance
    :return: (2N, 2N), kernel matrix
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    # (n_samples, C, H, W)
    total = torch.cat([source, target], dim=0)
    if not matrix_op:
        L2_distance = torch.zeros([total.size(0), total.size(0)], device=total.device)
        for i in range(total.size(0)):
            for j in range(total.size(0)):
                L2_distance[i, j] = ((total[i] - total[j])**2).sum()
    else:
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(3)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(3)))
        L2_distance = ((total0-total1)**2).view(int(total.size(0)), int(total.size(0)), -1).sum(-1)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def JAN(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)

    assert layer_num == len(kernel_muls), 'ERROR: layer_num not equals to len(kernel_muls)'
    assert layer_num == len(kernel_nums), 'ERROR: layer_num not equals to len(kernel_nums)'
    assert layer_num == len(fix_sigma_list), 'ERROR: layer_num not equals tot len(fix_sigma_list)'

    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma, matrix_op=True)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
        loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    return loss / float(batch_size)
