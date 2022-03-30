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
from torch.nn import init
from torchvision import io
from torch.utils.data import DataLoader
from dataset.dataset import NTIRE_Training_Dataset, NTIRE_Validation_Dataset
import models
import losses
from tqdm import tqdm
from utils import utils
from utils import metrics
import logging

from torch.utils.tensorboard import SummaryWriter
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

def train(abs_step, args, model, device, train_loader, optimizer, epoch, criterion, summary_writer):
    
    running_loss = 0.0
    smooth_loss = 0.0
    batches = 0
    losses = {key:0 for key in criterion._record.keys()}
    model.train()

    desc_phase = f"Training {epoch+1}/{args.epochs}" 
    tqbar = tqdm(train_loader, leave=False, total=len(train_loader), desc=desc_phase)
    for batch_idx, batch_data in enumerate(tqbar, 0):
        batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                             batch_data['input2'].to(device)
        label = batch_data['label'].to(device)
        optimizer.zero_grad()
        pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
        loss = criterion(pred, label)
        
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        running_loss += current_loss
        batches += 1
        smooth_loss = running_loss / batches
        tqbar.set_description(f"{desc_phase}: loss: {current_loss:.5f} | avg-loss: {smooth_loss:.5f}", refresh=True)
        
        abs_step += label.shape[0]

        # Current loss
        summary_writer.add_scalar(f'Train/Loss', current_loss, abs_step)
        for key, value in criterion._record.items():
            summary_writer.add_scalar(f'Train/{key}', value, abs_step)
            losses[key] += value

    ## Average Loss
    summary_writer.add_scalar(f'Train-avg/Loss', smooth_loss, epoch + 1)
    for key, value in losses.items():
        summary_writer.add_scalar(f'Train-avg/{key}', losses[key] / batches,  epoch + 1)

    return smooth_loss, abs_step


def validation(val_abs_step, args, model, device, data_loader, epoch, criterion, summary_writer):

    running_loss = 0.0
    smooth_loss = 0.0
    batches = 0
    losses = {key:0 for key in criterion._record.keys()}

    model.eval()
    n_samples = len(data_loader)
    avg_psnr = 0
    avg_mulaw = 0
    with torch.no_grad():
        desc_phase = f"Validation {epoch+1}/{args.epochs}" 
        tqbar = tqdm(data_loader, leave=False, total=len(data_loader), desc=desc_phase)
        for batch_idx, batch_data in enumerate(tqbar, 0):
            
            batch_ldr0 = batch_data['input0'].to(device)
            batch_ldr1 = batch_data['input1'].to(device)
            batch_ldr2 = batch_data['input2'].to(device)
            label = batch_data['label'].to(device)
            pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
            loss = criterion(pred, label)
            current_loss = loss.item()
            running_loss += current_loss
            batches += 1
            smooth_loss = running_loss / batches
            tqbar.set_description(f"{desc_phase}: loss: {current_loss:.5f} | avg-loss: {smooth_loss:.5f}", refresh=True)

            val_abs_step += label.shape[0]
            summary_writer.add_scalar(f'Valid/Loss', current_loss, val_abs_step)
            for key, value in criterion._record.items():
                summary_writer.add_scalar(f'Valid/{key}', value, val_abs_step)
                losses[key] += value
        
        summary_writer.add_scalar(f'Valid-avg/Loss', smooth_loss, epoch + 1)
        for key, value in losses.items():
            summary_writer.add_scalar(f'Valid-avg/{key}', losses[key] / batches,  epoch + 1)

    avg_psnr /= n_samples

    return smooth_loss, val_abs_step


def validation_visual(args, model, device, data_loader, epoch):
    model.eval()
    n_samples = len(data_loader)
    avg_psnr = 0
    avg_mulaw = 0
    c_samples = 0
    save_samples = 0
    save_path = os.path.join(args.logdir, "samples", str(epoch+1))
    if (epoch + 1) % args.save_samples_every == 0:
        save_samples = args.save_samples
        os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        desc_phase = f"Visual {epoch+1}/{args.epochs}" 
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

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.logdir,"models"), exist_ok=True)
    writer = SummaryWriter(args.logdir, filename_suffix="metrics")
    logging_file = args.logdir + "/output.log"
    handlers = [logging.FileHandler(logging_file), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)
    logging.info('## Start')
    logging.info(f" ".join(sys.argv))
    logging.info('__   Config   __')
    for arg in vars(args):
        logging.info(f'{arg} = {getattr(args, arg)}')

    # random seed
    if args.seed is not None:
        utils.set_random_seed(args.seed)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device('cpu')

    # dataset and dataloader
    train_dataset = NTIRE_Training_Dataset(opt=args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_dataset = NTIRE_Validation_Dataset(opt=args, crop_size=args.crop_train_data or 256) # same as tr size
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers // 2, pin_memory=True)

    visual_val_dataset = NTIRE_Validation_Dataset(opt=args, crop_size=args.crop_test_data)
    visual_val_loader = DataLoader(visual_val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


    # model architectures
    model = models.create_model(args)
    logging.info('## Arch:\n' + str(model) + '\n##')
    if args.init_kaiming:
        utils.init_parameters(model)
    
    model.to(device)

    stored = [0]
       
    # loss
    criterion = losses.create_loss(args)
    if criterion._requires_cuda:
        criterion.to(device)
    logging.info('## Loss arch:\n\n' + str(criterion) + '\n##')
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    

    first_epoch = 0
    abs_step = 0
    val_abs_step = 0
    if args.continue_training:
        model, optimizer, first_epoch, metrics, steps = utils.init_from_checkpoint(args, model, optimizer, device )
        stored[0] = metrics.get("best_mulaw", 0)
        abs_step, val_abs_step = steps
    
    if torch.cuda.device_count() > 1 and args.gpu is None:
        model = nn.DataParallel(model)


    all_, trainable_ = utils.model_parameters(model)
    if args.continue_training:
        logging.info('__   Continue Training   __')
    else:
        logging.info(f"Total params: {all_:,} | Trainable params: {trainable_:,}")
        logging.info('__      Start Training   __')
    
    for epoch in range(first_epoch, args.epochs):
        
        epoch_str = f"00{epoch+1}"[-3:]

        #lr = utils.simple_adjust_learning_rate(args, optimizer, epoch)
        lr = utils.linear_decay_lr(args, optimizer, epoch)
        writer.add_scalar('lr', lr, epoch + 1)

        
        # Train
        init = time.time()
        train_loss, abs_step = train(abs_step, args, model, device, train_loader, optimizer, epoch, criterion, writer)
        end = time.time()
        duration = utils.compute_timedelta(init, end)
        logging.info(f'[{epoch_str}] (+{duration}) | Train loss: {train_loss:.5f}')

        train_dataset.refresh_list()
        
        # Validation
        init = time.time()
        val_loss, val_abs_step = validation(val_abs_step, args, model, device, val_loader,  epoch, criterion, writer)
        end = time.time()
        duration = utils.compute_timedelta(init, end)
        logging.info(f'[{epoch_str}] (+{duration}) | Val loss  : {val_loss:.5f}')

        # Metrics
        init = time.time()
        val_psnr, val_mulaw = validation_visual(args, model, device, visual_val_loader, epoch)
        end = time.time()
        duration = utils.compute_timedelta(init, end)
        logging.info(f'[{epoch_str}] (+{duration}) | Val PSNR  : {val_psnr:.4f} | Val MuLaw: {val_mulaw:.4f}')
        writer.add_scalar('PSNR/L', val_psnr, epoch + 1)
        writer.add_scalar('PSNR/Mu', val_mulaw, epoch + 1)

        

        ## Store model and optimizer
        state_dict = model.state_dict()

        # if it is a dataparalle, unwrap first
        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()

        save_dict = {
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
            'steps':[abs_step, val_abs_step],
            'metrics':dict(best_mulaw=stored[0])}
        torch.save(save_dict, os.path.join(args.logdir,  "models",'latest_checkpoint.pth'))
        if val_mulaw > stored[0]:
            logging.info(f'[{epoch_str}] Store Best | Val PSNR  : {val_psnr:.4f} | Val MuLaw: {val_mulaw:.4f}')
            torch.save(save_dict, os.path.join(args.logdir,  "models",'best_checkpoint.pth'))
            stored[0] = val_mulaw
            with open(os.path.join(args.logdir, 'best_checkpoint.json'), 'w') as f:
                f.write('best epoch:' + str(epoch) + '\n')
                f.write('Val set: Average PSNR: {:.4f}, mu_law: {:.4f}\n'.format(val_psnr, val_mulaw))

        if (epoch + 1) % args.save_every == 0:
            torch.save(save_dict, os.path.join(args.logdir,  "models",f'{epoch+1}_checkpoint.pth'))

    final_metrics = {
        "hparam/weights":trainable_,
        "hparam/train_loss":train_loss, 
        "hparam/val_loss":val_loss, 
        "hparam/val_final_psnr": val_psnr,
        "hparam/val_final_mulaw": val_mulaw,
        "hparam/val_best_mulaw": stored[0]
    }

    utils.log_hparams(writer, args, final_metrics)
    logging.info("##END")


if __name__ == '__main__':
    main()
