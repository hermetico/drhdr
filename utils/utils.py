import os
import numpy as np
import cv2
import imageio
from math import log10
import datetime
import random
import torch
import torch.nn as nn
import torch.nn.init as init


def linear_decay_lr(args, optimizer, epoch):
    decay_after = args.lr_decay_after
    n_epochs = args.epochs
    mult =  1. if epoch < decay_after else 1 - float(epoch - decay_after) / (n_epochs - decay_after)
    lr = args.lr * mult
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


def init_from_checkpoint(args, model, optimizer, device):
    base_path = args.continue_from or args.logdir

    load_path = os.path.join(base_path,  "models",f'{args.start_epoch}_checkpoint.pth')
    if args.start_epoch == 0:
        load_path = os.path.join(base_path,  "models",'latest_checkpoint.pth')
        
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    first_epoch = checkpoint['epoch']
    steps = checkpoint.get("steps", [0,0])
    metrics = checkpoint.get("metrics", dict())
    # move optimizer stuff to device
    # https://github.com/pytorch/pytorch/issues/2830
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    return model, optimizer, first_epoch, metrics, steps


def load_from_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def set_random_seed(seed):
    """Set random seed for reproduce"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_timedelta(init, end):
    duration = end - init
    delta = datetime.timedelta(seconds=duration)
    delta_str = str(delta).split(".")[0]
    return delta_str


def model_parameters(model):
    all_ = sum(p.numel() for p in model.parameters())
    trainable_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_, trainable_


def log_hparams(writer, args, final_metrics):
    final_params = {arg:getattr(args, arg) for arg in vars(args)}
    writer.add_hparams(final_params, final_metrics)