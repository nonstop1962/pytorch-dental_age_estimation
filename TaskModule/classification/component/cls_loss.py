import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import ctc_loss
from util import get_loss

logger = logging.getLogger("Logger")


def cls_loss(cfg_loss):
    if cfg_loss is None:
        raise NotImplementedError("Loss")

    elif cfg_loss["name"] in key2loss:

        loss_name = cfg_loss["name"]
        loss_params = {k: v for k, v in cfg_loss.items() if k != "name"}

        logger.info(f'[{"LOSS".center(9)}] {loss_name} [params] {loss_params}')
        return functools.partial(key2loss[loss_name], **loss_params)

    else:
        logger.info(f'[{"LOSS".center(9)}] Try to use pre-defined loss from util/loss.py')
        return get_loss(cfg_loss)


def cross_entropy_cls(input, target):
    """
    input: outputs of the network
    target: label (just class value (0, 1, 2, 3, ..))
    """
    criterion = nn.CrossEntropyLoss()
    return criterion(input, target)


key2loss = {
    "cross_entropy_cls": cross_entropy_cls,
}
