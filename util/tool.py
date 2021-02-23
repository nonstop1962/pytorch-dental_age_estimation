import datetime
import logging
import os
import shutil

import yaml
from torch.utils.tensorboard import SummaryWriter


def configuration(log_root_dir, config, task):

    cfg = _read_yml(config)
    load_path = _check_config_validity(cfg, task)

    if load_path is None:
        log_root_dir, base_name = log_root_dir, os.path.basename(config)[:-4]
    else:
        loaded_weight_name = os.path.splitext(
            os.path.basename(cfg["module"]["load_state"]["file"])
        )[0]
        log_root_dir, base_name = load_path, f"eval_result_{loaded_weight_name}"

    x = datetime.datetime.now()
    run_id = f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"

    logdir = os.path.join(
        log_root_dir, base_name, str(cfg["setting"].get("seed", 1337)), str(run_id)
    )

    os.makedirs(logdir, exist_ok=True)
    shutil.copy(config, logdir)

    _get_logger(logdir)
    # TODO: validation에서 log 저장하고 싶을 땐 어떻게하지? 일단 아래와 같이 수정함
    # writer = get_writer(logdir) if task == "train" else None
    if task == "train" or task == "eval":
        writer = get_writer(logdir)
    else:
        None
    print(f"RUNDIR: {logdir}")

    return cfg["setting"], cfg["module"], writer


def _read_yml(config):
    with open(config, encoding="utf-8") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    cfg_import = cfg.pop("import", None)
    if cfg_import:
        cfg_mother = _read_yml(cfg_import)

        m_keys = list(cfg_mother.keys())
        for c_key in list(cfg.keys()):
            if c_key in m_keys:
                for k, v in cfg[c_key].items():
                    if type(cfg_mother[c_key][k]) == dict:
                        cfg_mother[c_key][k].update(cfg[c_key][k])
                    else:
                        cfg_mother[c_key][k] = cfg[c_key][k]
            else:
                cfg_mother[c_key] = cfg[c_key]

        cfg = cfg_mother

    return cfg


def _get_logger(logdir):
    logger = logging.getLogger("Logger")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, f"run_{ts}.log")
    hdlr = logging.FileHandler(file_path, mode="w")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.propagate = False

    return logger


def get_writer(logdir):
    writer = SummaryWriter(logdir)

    return writer


def get_meter():

    train_meter = meter()
    val_meter = meter()
    time_meter = averageMeter()

    return train_meter, val_meter, time_meter


class meter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.meters = {}

    def reset(self):
        for _, mtr in self.meters.items():
            mtr.reset()

    def update(self, loss):

        if type(loss) == dict:
            keys, loss = list(loss.keys()), list(loss.values())
        elif type(loss) == list:
            keys = ["loss" + str(i) for i in range(len(loss))]
        else:
            keys = ["loss"]
            loss = [loss]

        for key, l in zip(keys, loss):
            if key not in self.meters.keys():
                self.meters[key] = averageMeter()
            self.meters[key].update(l)


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _check_config_validity(cfg, task):
    """checks validity of config"""
    assert "module" in cfg
    assert "setting" in cfg

    module = cfg["module"]
    assert "mode" in module
    assert "network" in module
    assert "loss" in module
    assert "optimizer" in module
    assert "scheduler" in module
    assert "metric" in module
    assert "data" in module

    if task in ["val", "eval"]:
        assert "load_state" in module, "load model file for Evaluation Task"
        assert "file" in module["load_state"]
        load_path = os.path.dirname(module["load_state"]["file"])
    else:
        load_path = None

    data = module["data"]
    # assert 'srproj' in data
    assert "code" in data
    # assert 'training' in data
    # assert 'validation' in data

    # t, v = data['training'], data['validation']
    # assert 'batch_size' in t, v

    setting = cfg["setting"]
    assert "seed" in setting
    assert "print_time" in setting
    assert "train_iters" in setting
    assert "print_interval" in setting
    assert "val_interval" in setting
    assert "save_interval" in setting

    print("Config file validation passed")

    return load_path
