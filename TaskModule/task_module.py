import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# "get_" 는 util에 정의된 함수, "cls_" 는 classification module에 정의된 함수

from util import get_dataloader, get_optimizer, get_scheduler

# Logger
logger = logging.getLogger("Logger")


class task_module:
    def __init__(self, save_name, writer):

        # Basic Setting
        self.save_name = save_name
        self.save_path = writer.file_writer.get_logdir() if writer is not None else None
        self.writer = writer
        self.best_score = 0

        # Setup Model
        # Don't need to model.to(device)
        self.network = torch.nn.Module()

        # Setup Optimizer and Scheduler
        self.optimizer = torch.nn.Module()
        self.scheduler = torch.nn.Module()

    def _save_state(self, i_tr, score, best=False, additional={}):
        self.network.eval()

        state = {
            "iter": i_tr + 1,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "score": score,
        }

        state.update(additional)

        name = "best" if best else i_tr + 1
        save_path = os.path.join(
            self.save_path,
            f"{self.save_name}_{name}_model.pkl",
        )
        torch.save(state, save_path)

        logger.info(f"[SAVE] {self.save_name}_{name}_model")

    def _load_state(self, cfg_load):

        if cfg_load is not None:

            assert cfg_load["file"], "No State File Given"
            assert os.path.isfile(
                cfg_load["file"]
            ), f'No State File Found At { cfg_load["file"]}'

            state = torch.load(cfg_load["file"])

            if cfg_load.get("backbone_only", False):
                network_state_dict = self.network.state_dict()
                loaded_state_dict = self._convert_state(state["network_state"])
                # construct backbone only state_dict
                loaded_state_dict_backbone = {}
                for key in loaded_state_dict:
                    if "backbone" in key:
                        loaded_state_dict_backbone[key] = loaded_state_dict[key]
                # update network_state_dict with backbone only state_dict
                network_state_dict.update(loaded_state_dict_backbone)
                self.network.load_state_dict(network_state_dict)
            else:
                network_state = self._convert_state(
                    state.get("network_state", state.get("model_state"))
                )
                self.network.load_state_dict(self._convert_state(network_state))

            # loading specific layers sample
            # self.network.load_state_dict(self._convert_state(state["network_state"]))
            logger.info(
                f'[{"NETWORK".center(9)}] Trained Model Loaded from: {cfg_load["file"]}'
            )

            if cfg_load.get("optimizer", False):
                self.optimizer.load_state_dict(state["optimizer_state"])
                self.scheduler.load_state_dict(state["scheduler_state"])
                logger.info("Optimizer loaded")

            if cfg_load.get("iteration", False):
                self.start_iter = state["iter"] - 1
                logger.info(f"Start_iter loaded: {self.start_iter + 1}")

    def _convert_state(self, state):
        # erase '.module' for multi-gpu case
        state_modified = {}

        for k, v in state.items():
            split_k = k.split(".")
            name = ""

            for k_item in split_k:
                if k_item != "module" and k_item != "network":
                    name += k_item + "."

            state_modified[name[:-1]] = v

        return state_modified

    def _check_best_score(self, i_tr, score, key, additional={}):

        assert type(score) == dict, "Score should be passed to _check_bst_score as Dictionary"

        for k, v in score.items():
            logger.info(f"{k.rjust(18)}: {v}")
            if self.writer is not None and np.isscalar(v):
                self.writer.add_scalar(f"Metric/{k}", v, i_tr + 1)

        current_score = score[key]
        if current_score >= self.best_score:
            self.best_score = current_score

            if self.writer is not None:
                self._save_state(i_tr, current_score, best=True, additional=additional)

    def train_step(self, image, label, i):
        raise Exception("You called dummy 'train_step' function! Write your own 'train_step'")

    def validation_step(self, image, label):
        raise Exception(
            "You called dummy 'validation_step' function! Write your own 'validation_step'"
        )

    def evaluation_step(self, image, label):
        raise Exception(
            "You called dummy 'evaluation_step' function! Write your own 'evaluation_step'"
        )

    def on_save_interval(self, i):
        logger.info("[NOTHING SAVED] you didn't code 'on_save_interval' function")

    def on_validation_end(self, i):
        logger.info("[NOTHING HAPPENED] you didn't code 'on_validation_end' function")


if __name__ == "__main__":
    print(1)
