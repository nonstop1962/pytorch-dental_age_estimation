import logging

from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop

logger = logging.getLogger("Logger")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(opt_dict, model_params):
    optimizer = _get_optimizer_instance(opt_dict)

    params = {k: v for k, v in opt_dict.items() if k != "name"}

    optimizer = optimizer(model_params, **params)

    return optimizer


def _get_optimizer_instance(opt_dict):
    if opt_dict is None:
        logger.info("[OPTIMIZER] SGD")
        return SGD

    else:
        opt_name = opt_dict["name"]
        if opt_name not in key2opt:
            raise NotImplementedError(f"[OPTIMIZER] {opt_name} not implemented")

        logger.info(f"[OPTIMIZER] {opt_name}")
        return key2opt[opt_name]
