import collections
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import model_zoo


def load_pretrained_weights(model, model_name):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained weights for {model_name}")


def get_last_conv_channels(model):
    """WARNING: This will return the number of output channels of the last conv layer defined in `model`,
    and it could differ from the actual output channels of the model."""
    layer_type_list = [torch.nn.Conv2d, torch.nn.ConvTranspose2d]
    return [module for module in model.modules() if type(module) in layer_type_list][
        -1
    ].out_channels
