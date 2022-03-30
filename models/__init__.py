import importlib
import torch
import logging

def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_lib_name = "models." + model_name + "_model"
    modellib = importlib.import_module(model_lib_name)


    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, torch.nn.Module):
            model = cls

    if model is None:
        logging.error("In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (model_lib_name, target_model_name))
        exit(0)

    return model

def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    logging.info("# Model [%s] created" % (type(instance).__name__))

    return instance