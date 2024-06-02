import argparse  
import os
import sys
import traceback
import time
import warnings
import pickle
from collections import OrderedDict
import yaml
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
def import_class(model):
        mod,seq,clas=model.rpartition('.')
        __import__(mod)
        try:
            return getattr(sys.modules[mod],clas)
        except AttributeError:
            raise ImportError('Class %s cannot be found(%s)' %(clas,traceback.format_exception(*sys.exc_info()))) 

def get_parser(add_help=False):

    #region arguments yapf: disable
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser( add_help=add_help, description='IO Processor')

    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

    # processor
    #parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

    # visulize and debug
    #parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
    #parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    #parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
    #endregion yapf: enable

    return parser

def load_arg(argv=None):
    parser=get_parser()
    arg=parser.parse_args(argv)
    model=import_class(arg.model)
    print("这就是结果：\n",model)
    
load_arg()



    
