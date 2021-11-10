import numpy as np
import torch
import random
from datetime import datetime
import os
from collections import Counter
import json
import copy
import shutil
import socket


def set_device(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device


def set_output_folder(args):
    parent_folder = args.output_dir
    current_time = datetime.now().strftime('%Y%b%d_%H-%M-%S')
    args.output_dir = os.path.join(parent_folder, f"{args.output_tag}_{current_time}_{socket.gethostname()}")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # endif


def check_label_distribution(label_array):
    label_counter = Counter(label_array)
    print("================= label distribution =================")
    for label, num in sorted(label_counter.items(), key=lambda x: x[0]):
        print("{} -- {}".format(label, num))
    # endfor


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(args.seed)
    pass


def load_model(model, device, model_file_path):
    if device.type == "cpu":
        model.load_state_dict(torch.load(model_file_path, map_location="cpu"))
    elif device.type == "cuda":
        model.load_state_dict(torch.load(model_file_path, map_location="cuda"))
    else:
        raise SystemError("model file cannot be loaded to device: {}".format(device))
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    pass


def write_params(args):
    args_dict_tmp = vars(args)
    args_dict = copy.deepcopy(args_dict_tmp)
    args_dict["device"] = args_dict["device"].type

    config_file = os.path.join(args.output_dir, 'params.json')
    with open(config_file, 'w') as f:
        f.write(json.dumps(args_dict) + '\n')

    param_file = os.path.join(args.output_dir, "params.txt")
    with open(param_file, "w") as f:
        f.write("============ parameters ============\n")
        print("============ parameters =============")
        for k, v in args_dict.items():
            f.write("{}: {}\n".format(k, v))
            print("{}: {}".format(k, v))
        # endfor
        print("=====================================")
    # endwith


def move_log_file_to_output_directory(output_dir, file_path):
    shutil.move(file_path, output_dir)
    pass
