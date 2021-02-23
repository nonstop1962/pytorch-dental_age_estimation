import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from TaskModule.classification.component import cls_loss, cls_metric
from TaskModule.classification.component.cls_dataset import (
    cls_dataset_directory,
)
from TaskModule.classification.network import cls_network

# "get_" 는 util에 정의된 함수, "cls_" 는 classification module에 정의된 함수
from TaskModule.task_module import task_module
from util import get_dataloader, get_optimizer, get_scheduler


class CLS_module(task_module):
    def __init__(self, cfg_module, writer):
        super().__init__(save_name=f'{cfg_module["network"]["name"]}_cls', writer=writer)

        dataset_type = cfg_module["data"].get("dataset_type", "srproj")

        if dataset_type == "directory":
            # Setup Dataloader (put custom dataset such as "cls_dataset")
            self.trainloader = get_dataloader(
                "training", cfg_module["data"], cls_dataset_directory
            )
            self.valloader = get_dataloader(
                "validation", cfg_module["data"], cls_dataset_directory
            )
        assert (
            self.trainloader.dataset.n_classes == self.valloader.dataset.n_classes
        ), "train/val dataset n_classes missmatch"
        cfg_module["network"].update(n_classes=self.trainloader.dataset.n_classes)
        cfg_module["metric"].update(n_classes=self.trainloader.dataset.n_classes)

        # Define Module type (for external usage)
        self.module_type = "classification"
        # Setup Model
        # Don't need to model.to(device)
        self.network = cls_network(cfg_module["network"])

        # Setup Loss
        self.loss = cls_loss(cfg_module["loss"])

        # Setup Metric
        self.metric = cls_metric(cfg_module["metric"])

        # Setup Optimizer and Scheduler
        self.optimizer = get_optimizer(cfg_module["optimizer"], self.network.parameters())
        self.scheduler_name = (
            cfg_module["scheduler"]["name"] if cfg_module["scheduler"] else ""
        )
        self.scheduler = get_scheduler(cfg_module["scheduler"], self.optimizer)

        # Load State
        self._load_state(cfg_module["load_state"])

    def train_step(self, image, label, i):

        self.optimizer.zero_grad()

        # Process Network
        output = self.network(image)
        loss = self.loss(output, label)

        # Update Network Weights
        loss.backward()
        self.optimizer.step()

        # Update Scheduler In Case
        if self.scheduler_name != "on_plateau":
            self.scheduler.step()

        return loss.cpu().detach().numpy()

    def validation_step(self, image, label):

        # Process Network
        output = self.network(image)
        loss = self.loss(output, label)

        # Calculate metric and update meter
        pred = output.data.max(1)[1].cpu().numpy()
        gt = label.data.cpu().numpy()
        self.metric.update(gt, pred)

        # Derive Viewer Image
        image = None  # if return None for images, nothing will be shown in tensorboard

        return loss.cpu().detach().numpy(), image

    def evaluation_step(self, image, label):

        # Process Network
        output = self.network(image)
        loss = self.loss(output, label)

        # Calculate metric and update meter
        pred = output.data.max(1)[1].cpu().numpy()
        pred_logits = output.data.cpu().numpy()

        gt = label.data.cpu().numpy()
        self.metric.update(gt, pred)

        # Derive Viewer Image
        image = None  # if return None for images, nothing will be shown in tensorboard

        return loss.cpu().detach().numpy(), image, pred, pred_logits

    def on_save_interval(self, i_tr):
        self._save_state(i_tr, score=0)

    def on_validation_end(self, i_tr):
        # Calculate Metric Score
        score = self.metric.get_scores()

        # Some variables you want to print
        additional = {
            "test": self.best_score,
        }
        self._check_best_score(i_tr, score, "Overall Acc", additional=additional)

        self.metric.reset()


if __name__ == "__main__":
    print(1)
