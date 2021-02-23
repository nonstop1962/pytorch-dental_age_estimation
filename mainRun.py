import logging
import random
import time
import datetime

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
from util import get_meter, get_writer

logger = logging.getLogger("Logger")


class MainRun(object):
    def __init__(self, cfg_setting, writer):

        # Setup seeds
        np.random.seed(cfg_setting.get("seed", 1337))
        random.seed(cfg_setting.get("seed", 1337))
        torch.manual_seed(cfg_setting.get("seed", 1337))
        torch.cuda.manual_seed(cfg_setting.get("seed", 1337))
        torch.backends.cudnn.enabled = True

        # Setup Writer
        self.writer = writer

        # Setup Intervals
        self.train_iters = cfg_setting.get("train_iters", 1)
        self.print_interval = cfg_setting.get("print_interval", 1)
        self.val_interval = cfg_setting.get("val_interval", 1)
        self.save_interval = cfg_setting.get("save_interval", 1)

        # Setup Variables
        self.tensorboard = cfg_setting.get("tensorboard", False)
        self.tb_window = cfg_setting.get("tensorboard_window", [1200, 900])
        self.idx_show = None
        self.print_time = cfg_setting.get("print_time", False)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        logger.info(
            f'[{"INSPECT".center(9)}] [tensorboard] {self.tensorboard} [time] {self.print_time}'
        )

        # Setup Average Meters
        self.train_meter, self.val_meter, self.time_meter = get_meter()

        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def Run(self, module, task, multi_gpu=False):
        # Setup Device
        module.network.to(self.device)
        for state in module.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # Check if Using Multi-GPU
        if multi_gpu:
            module.network = torch.nn.DataParallel(module.network)

        logger.info(f'[{"RUN".center(9)}] [MainRun] [task: {task.upper()}]')

        # Setup Writer and Run right task
        if task == "train":
            self.train(module)
        elif task == "val":
            self.tensorboard = False
            self.validation(module)
        elif task == "eval":
            self.tensorboard = False
            self.evaluation(module)
        else:
            raise (f"Task {task} not available")

    def train(self, module):
        epoch_iter = int(len(module.trainloader.dataset) / module.trainloader.batch_size)
        self.print_interval = epoch_iter if self.print_interval == 0 else self.print_interval
        self.val_interval = epoch_iter if self.val_interval == 0 else self.val_interval
        self.save_interval = epoch_iter if self.save_interval == 0 else self.save_interval

        i_tr = module.start_iter if hasattr(module, "start_iter") else 0
        flag = True

        start_time = time.time()
        while i_tr <= self.train_iters and flag:

            if self.print_time:
                self.start.record()

            for images, labels in module.trainloader:

                # Step Preparation
                module.network.train()
                images, labels = self.to_device(images, labels)

                # Train Step
                loss = module.train_step(images, labels, i_tr)

                # Update Train Step Result
                self.train_meter.update(loss)

                if (i_tr + 1) % self.print_interval == 0:
                    elapsed_time = time.time() - start_time
                    elapsed_time = str(datetime.timedelta(seconds=elapsed_time))[:-7]

                    # Record Train Iters Result
                    self.record(
                        i_tr, self.train_meter, images=None, task="Train", time=elapsed_time
                    )

                if (i_tr + 1) % self.val_interval == 0 or (i_tr + 1) == self.train_iters:
                    self.validation(module, i_tr)

                if (i_tr + 1) % self.save_interval == 0 or (i_tr + 1) == self.train_iters:
                    module.on_save_interval(i_tr)

                    if self.print_time:
                        self.start.record()

                if (i_tr + 1) == self.train_iters:
                    flag = False
                    break

                i_tr += 1

        module.trainloader.stop()
        module.valloader.stop()

    def validation(self, module, i_tr=0):
        if len(module.valloader) == 0:
            self.tensorboard = False
            return

        # Setup Tensorboard Image Viewer
        if self.tensorboard and (self.idx_show is None):
            # Calculate Average Image Size
            img_width = []
            img_height = []
            with torch.no_grad():
                module.network.eval()
                for i_img, (images, labels) in tqdm(enumerate(module.valloader)):
                    # Step Preparation
                    images, labels = self.to_device(images, labels)
                    # Validation Step
                    loss, show = module.validation_step(images, labels)
                    if show is None or len(show) == 0:
                        break
                    if type(show) == list:
                        show = torch.cat(show, dim=-1)
                    img_s = show.shape[-2:]

                    img_width.append(img_s[1])
                    img_height.append(img_s[0])
            if show is None or len(show) == 0:
                self.tb_window = [1, 1]
                self.idx_show = []
            else:
                self.tb_window = [1, len(module.valloader)]
                self.idx_show = list(range(len(module.valloader)))
            module.metric.reset()

        with torch.no_grad():
            # Step Preparation
            module.network.eval()
            image_show = []

            if self.print_time:
                self.start.record()

            for i_val, (images, labels) in tqdm(
                enumerate(module.valloader), total=len(module.valloader)
            ):
                # Step Preparation
                images, labels = self.to_device(images, labels)

                # Validation Step
                loss, show = module.validation_step(images, labels)

                # Update Validation Step Result
                self.val_meter.update(loss)

                if (self.tensorboard) and (i_val in self.idx_show):
                    if show is not None or len(show) > 0:
                        if type(show) == list:
                            show = torch.cat(show, dim=-1)
                        image_show.append(show)

        # Record Validation Result
        self.record(i_tr, self.val_meter, images=image_show, task="Valid")
        module.on_validation_end(i_tr)

    def evaluation(self, module):
        if len(module.valloader) == 0:
            self.tensorboard = False
            return

        # Setup Tensorboard Image Viewer
        if self.tensorboard and (self.idx_show is None):
            # Calculate Average Image Size
            img_width = []
            img_height = []
            with torch.no_grad():
                module.network.eval()
                for i_img, (images, labels) in tqdm(enumerate(module.valloader)):
                    # Step Preparation
                    images, labels = self.to_device(images, labels)
                    # Validation Step
                    loss, show = module.validation_step(images, labels)
                    if show is None or len(show) == 0:
                        break
                    if type(show) == list:
                        show = torch.cat(show, dim=-1)
                    img_s = show.shape[-2:]

                    img_width.append(img_s[1])
                    img_height.append(img_s[0])
            if show is None or len(show) == 0:
                self.tb_window = [1, 1]
                self.idx_show = []
            else:
                self.tb_window = [1, len(module.valloader)]
                self.idx_show = list(range(len(module.valloader)))
            module.metric.reset()

        with torch.no_grad():
            # Step Preparation
            module.network.eval()
            image_show = []

            if self.print_time:
                self.start.record()

            for i_val, (images, labels) in tqdm(
                enumerate(module.valloader), total=len(module.valloader)
            ):
                # Step Preparation
                images, labels = self.to_device(images, labels)

                # Validation Step
                # loss, show = module.validation_step(images, labels)
                loss, show, pred, pred_logits = module.evaluation_step(images, labels)


                if module.module_type == "classification":
                    logdir = self.writer.file_writer.get_logdir()
                    save_pred_txt_dir = os.path.join(logdir, "prediction_class")
                    os.makedirs(save_pred_txt_dir, exist_ok=True)
                    cur_img_name = module.valloader.dataset.files[i_val]["name"]
                    save_pred_txt_path = os.path.join(save_pred_txt_dir, f"{cur_img_name}.txt")
                    with open(save_pred_txt_path, "w") as f:
                        f.write(str(pred[0]))

                    save_pred_logits_txt_dir = os.path.join(logdir, "prediction_logits")
                    os.makedirs(save_pred_logits_txt_dir, exist_ok=True)

                    save_pred_logits_txt_path = os.path.join(
                        save_pred_logits_txt_dir, f"{cur_img_name}.txt"
                    )
                    with open(save_pred_logits_txt_path, "w") as f:
                        f.write(str(pred_logits[0]))
                # Update Validation Step Result
                self.val_meter.update(loss)

                if (self.tensorboard) and (i_val in self.idx_show):
                    if show is not None or len(show) > 0:
                        if type(show) == list:
                            show = torch.cat(show, dim=-1)
                        image_show.append(show)

        # Record Validation Result
        # self.record(i_tr, self.val_meter, images=image_show, task="Valid")

        # module.on_validation_end(i_tr)

    def to_device(self, images, labels):
        if type(images) == list:
            images_out = []
            for img in images:
                images_out += [img.to(self.device) if torch.is_tensor(img) else img]
        elif type(images) == dict:
            images_out = {}
            for key, img in images.items():
                images_out.update({key: img.to(self.device) if torch.is_tensor(img) else img})
        else:
            images_out = images.to(self.device) if torch.is_tensor(images) else images

        if type(labels) == list:
            labels_out = []
            for lbl in labels:
                labels_out += [lbl.to(self.device) if torch.is_tensor(lbl) else lbl]
        elif type(labels) == dict:
            labels_out = {}
            for key, lbl in labels.items():
                labels_out.update({key: lbl.to(self.device) if torch.is_tensor(lbl) else lbl})
        else:
            labels_out = labels.to(self.device) if torch.is_tensor(labels) else labels

        return images_out, labels_out

    def record(self, i_tr, meter, images=[], task="Train", time=None):

        if self.print_time:
            torch.cuda.synchronize()
            self.end.record()

        print_str = (
            f"[Train] Iter [{i_tr + 1}/{self.train_iters}]"
            if task == "Train"
            else "[Validation] "
        )
        print_str += " ".join([f"[{k}: {v.avg:.8f}]" for k, v in meter.meters.items()])

        if time is not None:
            print_str += f" Elapsed time: {time}"

        logger.info(print_str)

        if self.tensorboard and self.writer is not None:
            for key, mtr in meter.meters.items():
                self.writer.add_scalar("Loss/" + task + "-" + key, mtr.avg, i_tr + 1)

            if images:
                data_width = np.max([item.shape[-1] for item in images])
                data_height = np.max([item.shape[-2] for item in images])
                images_pad = [
                    F.pad(
                        img if img.dim() == 4 else img[None],
                        (
                            0,
                            data_width - img.shape[-1],
                            0,
                            data_height - img.shape[-2],
                        ),
                    )
                    for img in images
                ]
                self.writer.add_image(
                    "Image",
                    make_grid(torch.cat(images_pad), nrow=self.tb_window[0]),
                    i_tr + 1,
                )

        meter.reset()

        if self.print_time:
            logger.info(f"[TIME] [{task}] {self.start.elapsed_time(self.end):.2f} ms")
            self.start.record()


if __name__ == "__main__":
    MainRun(1, 1)
