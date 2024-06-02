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

class mymodel():
    def __init__(self, work_dir, save_log=True, print_log=True):
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.cur_time = time.time()
        self.split_timer = {}
        self.pavi_logger = None
        self.session_file = None
        self.model_text = ''

    def import_class(model):
        mod,seq,clas=model.rpartition('.')
        __import__(mod)
        try:
            return getattr(sys.modules[mod],clas)
        except AttributeError:
            raise ImportError('Class %s cannot be found(%s)' %(clas,traceback.format_exception(*sys.exc_info()))) 

    