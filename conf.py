
import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH  = 'Standard'
_C.MODEL.ARCH2 = 'Standard'


# Choice of (source, norm, tent)
# - source: baseline without adaptation
# - norm: test-time normalization
# - tent: test-time entropy minimization (ours)
_C.MODEL.ADAPTATION = 'source'

# By default tent is online, with updates persisting across batches.
# To make adaptation episodic, and reset the model for each batch, choose True.
_C.MODEL.EPISODIC = False


_C.MODEL.CKPT_PATH = "."
_C.MODEL.SAVE_PATH = "."

_C.MODEL.EPS = 0.

_C.MODEL.LOSS = "polyloss"

_C.MODEL.DATASET = "cifar10"

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [3]

# Number of examples to evaluate (10000 for all samples in CIFAR-10)
_C.CORRUPTION.NUM_EX = 10000

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0
_C.OPTIM.TEMP = 1.0

# Set up in the config file (config.py)
_C.OPTIM.ADAPT = "ent"
_C.OPTIM.ADAPTIVE = False
_C.OPTIM.TBN = True
_C.OPTIM.UPDATE = True

# ------------------------------- Transformer options ------------------------- #
_C.TRANSFORMER = CfgNode()
_C.TRANSFORMER.INPUT_DIM = 16
_C.TRANSFORMER.N_HEADS = 4
_C.TRANSFORMER.DIM_FF = 16
_C.TRANSFORMER.ACTIVATION = "relu"
_C.TRANSFORMER.PROBE_LAYERS = 1


# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 128

_C.TEST.DATASET = "cifar10.1"





# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# ------------------------------- Attacking options --------------------------- #
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

_C.ATTACK = CfgNode()

_C.ATTACK.METHOD = "PGD"
# Attack methods

_C.ATTACK.SOURCE = 10
# Malicious Number 

_C.ATTACK.TARGET = 1
# targeted Number 

_C.ATTACK.EPS = 1.0
# L_inf bound 

_C.ATTACK.ALPHA = 0.00392157
# attack update rate 

_C.ATTACK.STEPS = 500
# attack steps  

_C.ATTACK.WHITE = True
# attack white box, model is known  

_C.ATTACK.ADAPTIVE = False
# our attack set it as false
 
_C.ATTACK.TARGETED = False
# the attack is targeted or not

_C.ATTACK.PER = 0.0
# stealthy targeted attack weight

_C.ATTACK.WEIGHT_P = 0.0
_C.ATTACK.DFPIROR = 0.0
_C.ATTACK.DFTESTPIROR = 0.0
_C.ATTACK.Layer = 0

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# ---------------------------------- Misc options --------------------------- #

# Optional description of a config
_C.DESC = ""

# Note that non-determinism is still present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Output directory
_C.SAVE_DIR = "./output/test"

# Data directory
_C.DATA_DIR = "../../dataset"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

_C.LOG_DIR = "./eval_results/tta"

# Log datetime
_C.LOG_TIME = ''

_C.DEBUG = False


# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    print(args.opts)
    cfg.merge_from_list(args.opts)
    
    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)


def load_cfg_fom_file(cfg_file="cfgs/tent.yaml"):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    # parser = argparse.ArgumentParser(description=description)
    # parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
    #                     help="Config file location")
    # parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
    #                     help="See conf.py for all options")
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    # args = parser.parse_args()

    merge_from_file(cfg_file)
    # cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
    #     datefmt="%y/%m/%d %H:%M:%S",
    #     handlers=[
    #         logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
    #         logging.StreamHandler()
    #     ])

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    # logger.info(
    #     "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    # logger.info(cfg)

