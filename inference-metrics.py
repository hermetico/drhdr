# -*- coding:utf-8 _*-
import torch
from options import BaseOptions
import models
from utils.complexity_metrics import get_gmacs_and_params, get_runtime


def main():
    args = BaseOptions().parse()

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        device = torch.device('cuda')
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device('cpu')

    # model architecture
    model = models.create_model(args)


    with torch.inference_mode():
        print("Running ops metrics")
        total_macs, total_params = get_gmacs_and_params(model, device, input_size=(1, 3, 6, 1060, 1900))
        mean_runtime = get_runtime(model, device, input_size=(1, 3, 6, 1060, 1900))


    print("runtime per image [s] : " + str(mean_runtime))
    print("number of operations [GMAcc] : " + str(total_macs))
    print("number of parameters  : " + str(total_params))

    

if __name__ == '__main__':
    main()
