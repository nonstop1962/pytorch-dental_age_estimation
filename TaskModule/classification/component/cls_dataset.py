import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from util import directory_dataset

import logging

logger = logging.getLogger("Logger")


class cls_dataset_directory(directory_dataset):
    def __init__(
        self,
        path,
        n_classes=2,
        split="train",
        label="pre-encoded",
        mode="Classification",
        gray=False,
        img_size=None,
        resize_factor=1,
        augmentation=None,
        collate="resize",
        **kwargs,
    ):
        super().__init__(
            path=path,
            n_classes=n_classes,
            split=split,
            label=label,
            mode=mode,
            gray=gray,
            img_size=img_size,
        )

        self.define_imgsize()

        if collate == "resize":
            self.collate_function = self.resize_collate
        elif collate == "padding":
            self.collate_function = self.padding_collate
        else:
            raise NotImplementedError(f"Collate {collate} not implemented")
        logger.info(f'[{"DATA".center(9)}] [CLS collate] {collate}')

    def define_imgsize(self):
        if self.img_size is None:

            # average sample images to derive img_size
            img_width = []
            img_height = []
            for index in range(len(self)):
                # Load Images and Text or Image Labels
                if self.on_memory:
                    img, lbl = self.on_memory[index]
                else:
                    img, lbl = self.load_data(index)

                img_s = img.size
                img_width.append(img_s[0])
                img_height.append(img_s[1])

                if index > 20:
                    break
            self.img_size = (
                int(np.mean(img_width) / self.resize_factor),
                int(np.mean(img_height) / self.resize_factor),
            )
        else:
            self.img_size = (
                int(self.img_size[0] / self.resize_factor),
                int(self.img_size[1] / self.resize_factor),
            )

        logger.info(f'[{"DATA".center(9)}] [image size] {self.img_size}')
