import importlib
import torch
import logging

def find_loss_using_name(loss_name):
    # Given the option --loss [loss_name],
    # the file "losses/[loss_name]_loss.py"
    # will be imported.
    loss_lib_name = "losses." + loss_name + "_loss"
    losslib = importlib.import_module(loss_lib_name)


    loss = None
    target_loss_name = loss_name
    for name, cls in losslib.__dict__.items():
        if name.lower() == target_loss_name.lower() \
           and issubclass(cls, torch.nn.Module):
            loss = cls

    if loss is None:
        logging.error("In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (model_lib_name, target_model_name))
        exit(0)

    return loss

def create_loss(opt):
    loss = find_loss_using_name(opt.loss)
    instance = loss(opt)
    logging.info("# Loss [%s] created" % (type(instance).__name__))

    return instance