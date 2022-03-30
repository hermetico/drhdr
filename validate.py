# -*- coding:utf-8 _*-
import os
import sys
import cv2
import time
import math
import numpy as np
from options import BaseOptions
import torch
import torch.nn as nn
from torchvision import io
from torch.utils.data import DataLoader
from dataset.dataset import NTIRE_Validation_Dataset
import models
import losses
from tqdm import tqdm
from utils import utils
from utils import metrics
import logging

from torchvision.transforms import functional as tf

def mu_tonemap(hdr_image, mu=5000):
    return torch.log(1 + mu * hdr_image) / math.log(1 + mu)


def save_toned(tensor, path):
    #norm tensor
    linear = tensor ** 2.24
    norm_perc = torch.quantile(linear.type(torch.float64), .99).item()
    mu_pred = metrics.tanh_norm_mu_tonemap(linear, norm_perc)
    io.write_png(tf.convert_image_dtype(mu_pred, torch.uint8),path)

def save_raw(tensor, path):
    image = tensor.permute(1,2,0).numpy()
    align_ratio = (2 ** 16 - 1) / image.max()
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(path, cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR))


def validation_visual(args, model, device, data_loader):
    model.eval()
    n_samples = len(data_loader)
    avg_psnr = 0
    avg_mulaw = 0
    c_samples = 0
    save_samples = 0
    save_path = os.path.join(args.logdir, "samples", "validation")
    os.makedirs(save_path, exist_ok=True)
    save_samples = args.save_samples
    with torch.no_grad():
        desc_phase = f"Validation"
        tqbar = tqdm(data_loader, leave=False, total=len(data_loader), desc=desc_phase)
        for batch_idx, batch_data in enumerate(tqbar, 0):
            c_samples +=1
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                                 batch_data['input2'].to(device)
            label = batch_data['label'].to(device)
            
            pred = model(batch_ldr0, batch_ldr1, batch_ldr2)

            if save_samples > 0:
                save_samples -= 1
                sample = pred.squeeze(0)
                id = batch_data["image_name"][0]
                sample_path = os.path.join(save_path, f"{id}.png")
                save_toned(sample.clone().cpu(), sample_path)
                
                sample_path = os.path.join(save_path, f"{id}_uint16.png")
                save_raw(sample.clone().cpu(), sample_path)
                
                label_path = os.path.join(save_path, f"{id}_label_uint16.png")
                save_raw(label.squeeze(0).clone().cpu(), label_path)
            
            psnr_pred = torch.squeeze(pred.clone())
            psnr_label = torch.squeeze(label.clone())
            psnr_pred = psnr_pred.data.cpu().numpy().astype(np.float32)#.clip(0, 100)
            psnr_label = psnr_label.data.cpu().numpy().astype(np.float32)
            psnr = metrics.normalized_psnr(psnr_label, psnr_pred, psnr_label.max())
            mu_law = metrics.psnr_tanh_norm_mu_tonemap(psnr_label, psnr_pred)
            avg_psnr += psnr
            avg_mulaw += mu_law
            tqbar.set_description(f"{desc_phase}: psnr: {psnr:.5f} | avg-psnr: {avg_psnr/c_samples:.5f}", refresh=True)

    avg_psnr /= n_samples
    avg_mulaw /= n_samples

    return avg_psnr, avg_mulaw


def main():
    # settings
    args = BaseOptions().parse()

    handlers = [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)
    logging.info('## Start')
    logging.info(f" ".join(sys.argv))
    logging.info('__   Config   __')
    for arg in vars(args):
        logging.info(f'{arg} = {getattr(args, arg)}')

    # random seed
    if args.seed is not None:
        utils.set_random_seed(args.seed)

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device('cpu')

    visual_val_dataset = NTIRE_Validation_Dataset(opt=args, crop_size=args.crop_test_data)
    visual_val_loader = DataLoader(visual_val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


    # model architectures
    model = models.create_model(args)
    logging.info('## Arch:\n' + str(model) + '\n##')
    if args.init_kaiming:
        utils.init_parameters(model)
    
    model.to(device)

    base_path = args.continue_from or args.logdir
    model_path = os.path.join(base_path,  "models",f'{args.start_epoch}_checkpoint.pth')
    if args.start_epoch == 0:
        model_path = os.path.join(base_path,  "models",'latest_checkpoint.pth')

    model = utils.load_from_checkpoint(model_path, model, device)



    # Metrics
    init = time.time()
    val_psnr, val_mulaw = validation_visual(args, model, device, visual_val_loader)
    end = time.time()
    duration = utils.compute_timedelta(init, end)
    logging.info(f'[validation] (+{duration}) | Val PSNR  : {val_psnr:.4f} | Val MuLaw: {val_mulaw:.4f}')

    logging.info("##END")


if __name__ == '__main__':
    main()
