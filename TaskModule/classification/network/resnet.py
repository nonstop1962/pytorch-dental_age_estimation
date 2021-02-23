import logging

import torch
import torch.nn as nn
from backbone import (
    get_last_conv_channels,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

logger = logging.getLogger("Logger")


class Resnet(nn.Module):
    def __init__(self, name="resnet18", n_classes=3, **configs):
        super(Resnet, self).__init__()

        resnet = _get_network_instance(name)
        self.backbone = resnet(**configs)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(get_last_conv_channels(self), n_classes, 1),
        )

    def forward(self, images):
        out = self.backbone(images)[-1]
        out = self.classifier(out)
        out = out.squeeze(-1).squeeze(-1)
        return out

    def load_state(self, state):
        # erase '.module' for multi-gpu case
        state_modified = {}

        for k, v in state.items():
            split_k = k.split(".")
            name = ""

            for k_item in split_k:
                if k_item != "module":
                    name += k_item + "."

            state_modified[name[:-1]] = v
        self.load_state_dict(state_modified)


def _get_network_instance(name):
    try:
        return {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
            "resnet152": resnet152,
        }[name]
    except:
        raise (f"Resnet {name} not available")
