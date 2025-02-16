import os
import copy
import random
import numpy as np
from collections import OrderedDict as OD
from collections import defaultdict as DD
from collections.abc import Iterable
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

def set_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_logger(path):
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    formatted_date_time = f"{year:04d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}.log"
    log_path_name = path+formatted_date_time
    return set_logger(log_path_name)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

# >>> Optimizer Stuff <<<
def set_optimizer(args, parameters):
    optimizer = torch.optim.Adam(
        parameters, lr=args.lr,
    )
    return optimizer

# Save hyperparameters to log files
def log_hyperparameters(args, logger):  
    logger.info("Hyperparameters:")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

# Print model structure and parameter information
def print_model_parameters(model, mylogger):
    param_info = [(name, param.size(), param.numel())
                  for name, param in model.named_parameters()]
    param_info.sort(key=lambda x: x[2], reverse=True)  # Sort by number of parameters
    mylogger.info("Model's state_dict:")
    for name, size, num_params in param_info:
        mylogger.info(
            f"Layer: {name} | Size: {size} | Total Parameters: {num_params}")
    mylogger.info(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

def log_gpu_memory_usage(device, mylogger, step=""):
    allocated = torch.cuda.memory_allocated(
        device=device) / (1024 * 1024)  
    reserved = torch.cuda.memory_reserved(
        device=device) / (1024 * 1024)  
    max_allocated = torch.cuda.max_memory_allocated(
        device=device) / (1024 * 1024)  
    max_reserved = torch.cuda.max_memory_reserved(
        device=device) / (1024 * 1024)  
    mylogger.info(f"{step} GPU Memory Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Max Allocated: {max_allocated:.2f} MB, Max Reserved: {max_reserved:.2f} MB")