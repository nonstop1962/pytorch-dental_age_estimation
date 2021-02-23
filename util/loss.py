import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Logger")


def get_loss(cfg_loss):
    if cfg_loss is None:
        logger.info(f'[{"LOSS".center(9)}] default cross entropy')
        return cross_entropy2d

    else:
        loss_name = cfg_loss["name"]
        loss_params = {k: v for k, v in cfg_loss.items() if k != "name"}

        if loss_name not in key2loss:
            raise NotImplementedError(f"Loss {loss_name} not implemented")

        logger.info(f'[{"LOSS".center(9)}] {loss_name} [params] {loss_params}')
        return functools.partial(key2loss[loss_name], **loss_params)


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


key2loss = {
    "cross_entropy": cross_entropy2d,
}
