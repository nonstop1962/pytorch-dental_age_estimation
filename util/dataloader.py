import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data

from util.augmentation import get_augmentation

split_dict = {
    "training": "Training",
    "validation": "Validation",
}


def get_dataloader(split, cfg_data, dataset_object):
    """
    constructs DataLoader
    cfg: config dictionary
    dataset: dataset object
    split: key to cfg.
    """

    # Derive path
    cfg_data.update({"path": _path_interpreter(cfg_data["code"])})

    # Define Config
    cfg_split = cfg_data.get(split, {})

    # Setup Augmentation
    cfg_aug = cfg_split.get("augmentation", None)
    data_aug = get_augmentation(cfg_aug)

    # Setup Dataset
    data_split = cfg_split.get("split", split_dict[split])
    dataset = dataset_object(split=data_split, augmentation=data_aug, **cfg_data)

    # Setup Dataloader
    batch_size = cfg_split.get("batch_size", 1)
    num_workers = cfg_split.get("n_workers", 0)
    data.DataLoader.stop = _stop
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=cfg_split.get("shuffle", False),
        collate_fn=getattr(dataset, "collate_function", None),
        pin_memory=False,
        drop_last=True,
    )

    return loader


# monkey patch
def _stop(self):
    pass


def _path_interpreter(code):
    path = None
    if os.path.isdir(code) or os.path.isfile(code):
        # correct directory/file
        path = code

    return path
