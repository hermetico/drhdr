# -*- coding:utf-8 _*-
import os
import math
import torch
from options import BaseOptions
from torch.utils.data import DataLoader
from dataset.dataset import NTIRE_Inference_Dataset
import models
from tqdm import tqdm
from utils import utils
from utils import data_io
from torchvision import io
from torchvision.transforms import functional as tf
from utils.complexity_metrics import get_gmacs_and_params, get_runtime

def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = torch.tanh(hdr_image / norm_value)
    return mu_tonemap(bounded_hdr, mu)


def mu_tonemap(hdr_image, mu=5000):
    return torch.log(1 + mu * hdr_image) / math.log(1 + mu)


def save_raw(tensor, path):
    #norm tensor
    linear = tensor ** 2.24
    norm_perc = torch.quantile(linear.type(torch.float64), .99).item()
    mu_pred = tanh_norm_mu_tonemap(linear, norm_perc)
    #tf.to_pil_image(mu_pred).save(path)
    io.write_png(tf.convert_image_dtype(mu_pred, torch.uint8),path)

def main():
    # settings
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

    checkpoint_folder = args.logdir
    # dataset and dataloader
    inference_dataset = NTIRE_Inference_Dataset(opt=args, path=args.inference_test_dataset_dir)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=4)

    # model architectures
    model = models.create_model(args)
    model.to(device)
    model = utils.load_from_checkpoint(os.path.join(checkpoint_folder,  "models",'best_checkpoint.pth'), model, device)
    output_folder = os.path.join(checkpoint_folder, "inference")
    output_tone = os.path.join(checkpoint_folder, "tone")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_tone, exist_ok=True)

    n_samples = len(inference_loader)
    with torch.inference_mode():
        desc_phase = "" 
        tqbar = tqdm(inference_loader, leave=False, total=n_samples, desc=desc_phase)
        for batch_idx, batch_data in enumerate(tqbar, 0):
            id = batch_data["image_name"][0]
            name = id + ".png"
            ratio = id + "_alignratio.npy"
            
            img_output_path = os.path.join(output_folder, name)
            align_output_path = os.path.join(output_folder, ratio)
            tone_output_patch = os.path.join(output_tone, name)

            tqbar.set_description(name, refresh=True)

            batch_ldr0 = batch_data['input0'].to(device)
            batch_ldr1 = batch_data['input1'].to(device)
            batch_ldr2 = batch_data['input2'].to(device)


            output = model(batch_ldr0, batch_ldr1, batch_ldr2).squeeze(0).cpu()
            #output = torch.clamp(output, 0, 100)
            save_raw(output, tone_output_patch)
            # Reshape for numpy
            result = output.permute(1,2,0).numpy()
            data_io.imwrite_uint16_png(img_output_path, result, align_output_path)

    print("Running ops metrics")
    with torch.inference_mode():
        total_macs, total_params = get_gmacs_and_params(model, device, input_size=(1, 3, 6, 1060, 1900))
        mean_runtime = get_runtime(model, device, input_size=(1, 3, 6, 1060, 1900))


    print("runtime per image [s] : " + str(mean_runtime))
    print("number of operations [GMAcc] : " + str(total_macs))
    print("number of parameters  : " + str(total_params))

    metrics_path = os.path.join(output_folder, "readme.txt")
    with open(metrics_path, 'w') as f:
        f.write("runtime per image [s] : " + str(mean_runtime))
        f.write('\n')
        f.write("number of operations [GMAcc] : " + str(total_macs))
        f.write('\n')
        f.write("number of parameters  : " + str(total_params))
        f.write('\n')
        f.write("Other description: We have a Python/Pytorch implementation, and report single GPU runtime. The method was trained on the training dataset (- 250 for validation) for 300 epochs on random crops of 256x256px ")


if __name__ == '__main__':
    main()
