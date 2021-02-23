import torch.nn as nn
from TaskModule.classification import CLS_module


def taskModule(cfg_module, writer):

    module = _get_module_instance(cfg_module["mode"])
    module = module(cfg_module, writer)

    return module


def _get_module_instance(mode):
    try:
        return {
            "classification": CLS_module,
        }[mode]
    except:
        raise (f"Module {mode} not available")
