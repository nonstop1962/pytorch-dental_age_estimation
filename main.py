import argparse

from TaskModule import taskModule
from util import configuration

from mainRun import MainRun


def main(args):
    # Read yaml Setting File
    cfg_setting, cfg_module, writer = configuration(
        args["log_root_dir"], args["config"], args["task"]
    )

    # Build mainRun
    mainRun = MainRun(cfg_setting, writer)

    # Build Module
    Module = taskModule(cfg_module, writer)

    # Run
    mainRun.Run(Module, args["task"], args["multi_gpu"])

    # Writer returns tensorboard writer when task is 'train'
    # Writer returns     None    writer when task is  'val'
    return writer


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
        "--task",
        "-t",
        nargs="?",
        type=str,
        default="train",
        help="Choose task to run 'train/val",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        default=False,
        help="Use multi-gpu for the task",
    )
    args = parser.parse_args()

    writer = main(vars(args))

    # When task is 'train', directory guide is printed
    if writer:
        print(
            f'[{"COMPLETE".center(9)}] Please check Training result directory: {writer.file_writer.get_logdir()}'
        )
