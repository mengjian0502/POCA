import argparse
import os
import torch as th

class AverageMeter(object):
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


class DataSaverHook:
    def __init__(self, store_input=True, store_output=True) -> None:
        self.store_input = store_input
        self.store_output = store_output

        self.input = None
        self.output = None
    
    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input = input_batch
        
        if self.store_output:
            self.output = output_batch


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')