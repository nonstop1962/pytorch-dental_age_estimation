import os
import sys
from os.path import join as pjoin

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import logging

from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms


logger = logging.getLogger("Logger")


class base_dataset(data.Dataset):
    def __init__(
        self,
        mode=None,
        gray=False,
        img_size=None,
        resize_factor=1,
        augmentation=None,
        collate=None,
        **kwargs,
    ):
        # Construct with Zero files
        self.files = []

        # Config other options
        self.mode = mode
        self.gray = gray

        self.img_size = img_size
        self.resize_factor = resize_factor
        self.compose_aug = augmentation

        self.to_tensor = transforms.ToTensor()

        if collate is not None:
            assert collate in ["resize", "padding"]
            self.collate_function = {
                "resize": self.resize_collate,
                "padding": self.padding_collate,
            }[collate]
            logger.info(f'[{"DATA".center(9)}] [Base dataset collate] {collate}')

    def __len__(self):
        return len(self.files)

    def stack_on_memory(self):
        on_memory = []
        for idx in range(len(self)):
            on_memory.append(self.load_data(idx))

        logger.info(f'[{"DATA".center(9)}] [ON_MEMORY] activated')

        return on_memory

    def __getitem__(self, index):

        # Load Images and Text or Image Labels
        if self.on_memory:
            img, lbl = self.on_memory[index]
        else:
            img, lbl = self.load_data(index)

        img, lbl = self.process(img, lbl)

        return img, lbl

    def load_data(self, index):
        return

    def process(self, img, lbl):
        # Resize Image and Label
        img, lbl = self.resize(img, lbl)

        """ Dummy function for user customization """
        img, lbl = self.btw_resize_aug(img, lbl)

        # Process augmentation defined in util/augmentation.py
        img, lbl = self.augmentation(img, lbl)

        """ Dummy function for user customization """
        img, lbl = self.btw_aug_transform(img, lbl)

        # Transform data to torch tensor
        img, lbl = self.transform(img, lbl)

        return img, lbl

    def resize(self, img, lbl):
        # 인풋으로 받은 img_size 우선시,
        # img_size가 없을 경우 resize_factor에 따라 크기 조정
        org_size = img.size
        img_size = (
            self.img_size
            if self.img_size is not None
            else (
                int(org_size[0] / self.resize_factor),
                int(org_size[1] / self.resize_factor),
            )
        )

        if (img_size[0] == org_size[0]) and (img_size[1] == org_size[1]):
            return img, lbl

        img = img.resize(img_size)

        return img, lbl

    def augmentation(self, img, lbl):

        if self.compose_aug is not None:
            if self.mode in ["Classification"]:
                img = self.compose_aug(img)
            else:
                img, lbl = self.compose_aug(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = self.to_tensor(img)

        if self.mode in ["Classification"]:
            lbl = torch.tensor(lbl).long()

        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        return img, lbl

    def btw_resize_aug(self, img, lbl):
        return img, lbl

    def btw_aug_transform(self, img, lbl):
        return img, lbl

    def resize_collate(self, batch):
        img_out = []
        lbl_out = []

        img_size = (
            (self.img_size[1], self.img_size[0])
            if self.img_size is not None
            else (
                int(np.mean([img.shape[-2] for img, _ in batch])),
                int(np.mean([img.shape[-1] for img, _ in batch])),
            )
        )

        # 이미 tensor 형태로 들어온 상태기 때문에 F.interpolate 함수 사용해보자
        for img, lbl in batch:
            img_out += [F.interpolate(img[None], (img_size[0], img_size[1]))]

            if self.mode in ["Classification"]:
                lbl_out += [lbl.unsqueeze(0)]

            else:
                raise NotImplementedError(f"Mode {self.mode} not implemented")

        img = torch.cat(img_out)
        lbl = torch.cat(lbl_out)

        return img, lbl

    def padding_collate(self, batch):
        height = [item[0].shape[-2] for item in batch]
        width = [item[0].shape[-1] for item in batch]

        max_h, max_w = np.max(height), np.max(width)

        img_out = []
        lbl_out = []
        for img, lbl in batch:
            img_out += [F.pad(img[None], (0, max_w - img.shape[-1], 0, max_h - img.shape[-2]))]
            if self.mode in ["Classification"]:
                lbl_out += [lbl.unsqueeze(0)]
            else:
                raise NotImplementedError(f"Mode {self.mode} not implemented")

        img = torch.cat(img_out)
        lbl = torch.cat(lbl_out)

        return img, lbl


class directory_dataset(base_dataset):
    def __init__(
        self,
        path,
        mode="Classification",
        n_classes=2,
        two_class=False,
        split="train_cls",
        label="binary",
        on_memory=False,
        gray=False,
        img_size=None,
        resize_factor=1,
        augmentation=None,
        **kwargs,
    ):
        super().__init__(
            mode=mode,
            gray=gray,
            img_size=img_size,
            resize_factor=resize_factor,
            augmentation=augmentation,
        )

        self.n_classes = n_classes

        self.files = self.read_directory(path, split, label)

        self.on_memory = self.stack_on_memory() if on_memory else []

    def read_directory(self, path, split, label):
        with open(
            pjoin(path, "ImageSets", self.mode, split + ".txt"), "r", encoding="utf-8"
        ) as f:
            file_list = [id_.rstrip() for id_ in f]

        img_files = {
            os.path.splitext(fname)[0]: pjoin(path, "JPEGImages", fname)
            for fname in os.listdir(pjoin(path, "JPEGImages"))
        }
        lbl_files = {
            os.path.splitext(fname)[0]: pjoin(path, self.mode + "Class", label, fname)
            for fname in os.listdir(pjoin(path, self.mode + "Class", label))
        }

        files = [{"name": l, "image": img_files[l], "label": lbl_files[l]} for l in file_list]

        return files

    def load_data(self, index):
        # find file name
        img_path = self.files[index]["image"]
        lbl_path = self.files[index]["label"]

        # open image
        img = Image.open(img_path)
        if (
            len(img.getbands()) != 3
        ) and not self.gray:  # img is a grey-scale or have alpha channel
            img = img.convert("RGB")
        if (len(img.getbands()) == 3) and self.gray:
            img = img.getchannel("R")  # select the first channel

        if self.mode == "Classification":
            f_lbl = open(lbl_path, "r")
            lbl = int(f_lbl.read())

            if self.n_classes == 2:
                lbl = 1 if (lbl > 1) else lbl

        return img, lbl