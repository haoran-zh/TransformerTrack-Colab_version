import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import ltr.admin.settings as ws_settings


def run_training(train_module, train_name, cudnn_benchmark=True):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))
    # train_module: dimp
    # train_name: transformer_dimp

    settings = ws_settings.Settings()  # initial the settings for the training.

    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'ltr/{}/{}'.format(train_module, train_name)  # create a new folder for the project
    print(settings.project_path)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))  # get
    # now we can use the imported function ltr.train_settings.train_module.train_name as expr_module(as a function)
    # settings for the chosen model

    expr_func = getattr(expr_module, 'run')  # get the specific function: run

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')

    args = parser.parse_args()
    # python run_training.py dimp transformer_dimp
    # args.train_module = dimp
    # args.train_name = transformer_dimp
    run_training(args.train_module, args.train_name, args.cudnn_benchmark)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)  # set process
    main()
