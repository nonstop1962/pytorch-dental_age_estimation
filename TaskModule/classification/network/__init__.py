import logging

from .resnet import Resnet

logger = logging.getLogger("Logger")


def cls_network(cfg_network):

    name = cfg_network["name"]

    model = _get_network_instance(name)

    model = model(**cfg_network)

    logger.info(f'[{"NETWORK".center(9)}] [name] {name} [params] {cfg_network}')

    return model


def _get_network_instance(name):
    try:
        return {
            "resnet18": Resnet,
            "resnet34": Resnet,
            "resnet50": Resnet,
            "resnet101": Resnet,
            "resnet152": Resnet,
        }[name]
    except:
        raise (f"Model {name} not available")
