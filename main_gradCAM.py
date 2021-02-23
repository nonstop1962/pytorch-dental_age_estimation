import torch
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms

from TaskModule import taskModule

from util.gradcam_resnet import *

import yaml
import os


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")

    parser.add_argument(
        "--log_root_dir",
        "-l",
        nargs="?",
        type=str,
        default="../stack_cell_results_0112",
        help="directory for saving results",
    )
    parser.add_argument(
        "--config",
        "-c",
        nargs="?",
        type=str,
        default="test/config/sample_font.yml",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--image-path", type=str, default="./examples/both.png", help="Input image path"
    )
    parser.add_argument(
        "--label-path", type=str, default="./examples/both.png", help="Input label path"
    )
    args = parser.parse_args()

    """python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization."""

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    device = get_device(True)
    # Model from torchvision
    module = taskModule(cfg["module"], writer=None)
    model = module.network.to(device)

    n_classes = cfg["module"]["network"]["n_classes"]
    module._load_state(cfg["module"]["load_state"])

    model.eval()

    # model = models.resnet50(pretrained=True)
    grad_cam = GradCam(
        model=model,
        feature_module=model.backbone.blocks[3],
        target_layer_names=["2"],
        use_cuda=True,
    )

    img = cv2.imread(args.image_path, 1)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    grayscale_cam = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    cv2.imwrite(os.path.join(args.log_root_dir, "cam.jpg"), cam)
