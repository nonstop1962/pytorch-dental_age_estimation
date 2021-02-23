import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def resnet18(**param_dict):
    return resnet(models.resnet18, **param_dict)


def resnet34(**param_dict):
    return resnet(models.resnet34, **param_dict)


def resnet50(**param_dict):
    return resnet(models.resnet50, **param_dict)


def resnet101(**param_dict):
    return resnet(models.resnet101, **param_dict)


def resnet152(**param_dict):
    return resnet(models.resnet152, **param_dict)


class resnet(nn.Module):
    def __init__(
        self, model, pretrained=True, n_channels=3, batchnorm_transfer_only_weight=True
    ):
        super(resnet, self).__init__()

        resnet_model = model(pretrained=pretrained)

        if n_channels != 3:
            self.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool

        self.blocks = nn.ModuleList()
        self.blocks.append(resnet_model.layer1)
        self.blocks.append(resnet_model.layer2)
        self.blocks.append(resnet_model.layer3)
        self.blocks.append(resnet_model.layer4)

        if batchnorm_transfer_only_weight:
            for layer in [
                module for module in self.modules() if type(module) != nn.Sequential
            ]:
                if layer.__class__ == nn.BatchNorm2d:
                    pass
                    layer.reset_running_stats()
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)

        return features
