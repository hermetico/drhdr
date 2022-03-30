import random
import torch
from torchvision.transforms import CenterCrop
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
from utils import data_io
import os.path as osp
from pathlib import Path
from torchvision import io
from torchvision.transforms import functional as tf


GT = "_gt.png"
SHORT = "_short.png"
MEDIUM = "_medium.png"
LONG = "_long.png"
ALIGN ="_alignratio.npy"
EXPOSURE ="_exposures.npy"
KEY = LONG

def sort_key(filepath):
    # sorts by id
    return  int(filepath.name.replace(KEY, ""))

def read_image(path):
    return tf.convert_image_dtype(io.read_image(path), torch.float32)

def read_images(paths):
    return [read_image(path) for path in paths]

def ev_alignment(img, expo, gamma):
    return ((img ** gamma) * 2.0**(-1*expo))**(1/gamma)

def cv2_read_images(fileNames):
    imgs = []
    for imgStr in fileNames:
        img=cv2.cvtColor(cv2.imread(imgStr, cv2.IMREAD_UNCHANGED),
                     cv2.COLOR_BGR2RGB) / 255.0
        imgs.append(img)
    return np.array(imgs)

class Augmentations:

    def pick(self):
        options = [self.hflip, self.vflip, self.rotate_half, self.rotate_all, self.nothing]
        return random.choice(options)

    def hflip(self, tensor):
        return tf.hflip(tensor)

    def vflip(self, tensor):
        return tf.vflip(tensor)

    def rotate_all(self, tensor):
        return tf.rotate(tensor, 180)

    def rotate_half(self, tensor):
        return tf.rotate(tensor, 90)

    def nothing(self, tensor):
        return tensor

class RandomCrop(torch.nn.Module):

    def __init__(self, size):
        
        super().__init__()
        self.size = size    # crop size
        self.i = 0
        self.j = 0

        if isinstance(size, int):
            self.size = (size, size)

    def update_coordinates(self, image):

        w, h = image.shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            self.i, self.j = 0, 0
        else:
            self.i = torch.randint(0, h - th + 1, size=(1, )).item()
            self.j = torch.randint(0, w - tw + 1, size=(1, )).item()

    def forward(self, img):
        h, w = self.size
        i, j = self.i, self.j
        return tf.crop(img, i, j, h, w)


class NTIRE_Training_Dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        root_dir = opt.dataset_dir
        self.rcrop = None
        if opt.crop_train_data is not None:
            self.rcrop = RandomCrop(opt.crop_train_data)

        self.augm = None 
        if not opt.not_augm:
            self.augm = Augmentations()

        if "~/" in root_dir:
            root_dir = root_dir.replace("~", os.path.expanduser("~"))
        root_dir = osp.join(root_dir, "train")
        scene_list = sorted(os.listdir(root_dir))
        self.all_list = []
        for id in scene_list:
            scene_path = Path(osp.join(root_dir, id))
            
            align_ratio_path = (scene_path / 'alignratio.npy').as_posix()
            exposure_path = (scene_path / 'exposures.npy').as_posix()

            scene_files = scene_path.glob("*"+KEY)
            scene_files = list(sorted(scene_files, key=sort_key))

            for file in scene_files:
                file = file.as_posix()
                imgs = []
                for _type in [SHORT, MEDIUM, LONG]:
                     imgs.append(file.replace(KEY, _type))
                label = file.replace(KEY, GT)

                self.all_list.append(dict(input=imgs, label=label, exposure=exposure_path, align=align_ratio_path))


        self.refresh_list()
        
        
    def refresh_list(self):
        limit = self.opt.limit_training_dataset
        if limit < 1:
            qty = int(len(self.all_list) * limit)
            self.patch_list = random.choices(self.all_list, k=qty)
        else:
            self.patch_list = self.all_list[:]


    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = torch.from_numpy(np.load(self.patch_list[index]["exposure"]))
        floating_exposures = exposures - exposures[1]

        # Read LDR images
        ldr_images = read_images(self.patch_list[index]["input"])
        # Read HDR label
        label = data_io.imread_uint16_png(self.patch_list[index]["label"], self.patch_list[index]["align"])

        # ldr images process
        s_gamma = 2.24
        if random.random() < 0.3:
            s_gamma += (random.random() * 0.2 - 0.1)

        image_short = ev_alignment(ldr_images[0], floating_exposures[0], s_gamma)
        image_medium = ldr_images[1]
        image_long = ev_alignment(ldr_images[2], floating_exposures[2], s_gamma)

        img0 = torch.cat((ldr_images[0], image_short), dim=0)
        img1 = torch.cat((ldr_images[1], image_medium), dim=0)
        img2 = torch.cat((ldr_images[2], image_long), dim=0)
        label = torch.from_numpy(label.astype(np.float32).transpose(2, 0, 1))
        
        # If center crop exists
        if self.rcrop is not None:
            self.rcrop.update_coordinates(img0)
            img0 = self.rcrop(img0)
            img1 = self.rcrop(img1)
            img2 = self.rcrop(img2)
            label = self.rcrop(label)
        
        if self.augm is not None:
            augmentation = self.augm.pick()

            img0 = augmentation(img0)
            img1 = augmentation(img1)
            img2 = augmentation(img2)
            label = augmentation(label)

        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label}
        return sample

    def __len__(self):
        return len(self.patch_list)


class NTIRE_Validation_Dataset(Dataset):

    def __init__(self, opt, mode="valid", crop_size=None):
        self.opt = opt
        root_dir = opt.dataset_dir
        self.ccrop = None
        if crop_size is not None:
            self.ccrop = CenterCrop(crop_size)

        if "~/" in root_dir:
            root_dir = root_dir.replace("~", os.path.expanduser("~"))
        root_dir = osp.join(root_dir, mode)
        subfolders = os.listdir(root_dir)
        # folder names are "003", "057"... can be sorted as regular strings
        subfolders = sorted(subfolders) 

        self.file_list = []
        for f in subfolders:
            folder = os.path.join(root_dir, f)
            
            label = os.path.join(folder, GT.lstrip("_"))
            align = os.path.join(folder, ALIGN.lstrip("_"))
            exposure = os.path.join(folder, EXPOSURE.lstrip("_"))
            imgs = []
            for _type in [SHORT, MEDIUM, LONG]:
                file = os.path.join(folder, _type.lstrip("_"))
                imgs.append(file)
            
            self.file_list.append(dict(input=imgs, label=label, exposure=exposure, align=align))

        limit = int(len(self.file_list) * opt.limit_validation_dataset)
        self.file_list = self.file_list[:limit]

    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = np.load(self.file_list[index]["exposure"])
        floating_exposures = exposures - exposures[1]

        # Read LDR images
        ldr_images = read_images(self.file_list[index]["input"])
        # Read HDR label
        label = data_io.imread_uint16_png(self.file_list[index]["label"], self.file_list[index]["align"])
        
        image_id = self.file_list[index]["label"].split('/')[-2]
        # ldr images process
        image_short = ev_alignment(ldr_images[0], floating_exposures[0], 2.24)
        image_medium = ldr_images[1]
        image_long = ev_alignment(ldr_images[2], floating_exposures[2], 2.24)

        img0 = torch.cat((ldr_images[0], image_short), dim=0)
        img1 = torch.cat((ldr_images[1], image_medium), dim=0)
        img2 = torch.cat((ldr_images[2], image_long), dim=0)
        label = torch.from_numpy(label.astype(np.float32).transpose(2, 0, 1))
        
        # If center crop exists
        if self.ccrop is not None:
            img0 = self.ccrop(img0)
            img1 = self.ccrop(img1)
            img2 = self.ccrop(img2)
            label = self.ccrop(label)

        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label, 'image_name': image_id}
        return sample

    def __len__(self):
        return len(self.file_list)


class NTIRE_Inference_Dataset(Dataset):

    def __init__(self, opt, path=False):
        self.opt = opt
        root_dir = path

        if "~/" in root_dir:
            root_dir = root_dir.replace("~", os.path.expanduser("~"))
        
        root_dir = Path(root_dir).glob("*" + KEY)
        paths = list(sorted(root_dir, key=sort_key))

        self.file_list = []
        for lp in paths:
            lp = lp.as_posix()
            exposure = lp.replace(KEY, EXPOSURE)
            
            imgs = []
            for _type in [SHORT, MEDIUM, LONG]:
                file = lp.replace(KEY, _type)
                imgs.append(file)
            
            
            self.file_list.append(dict(input=imgs, exposure=exposure))


    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = np.load(self.file_list[index]["exposure"])
        floating_exposures = exposures - exposures[1]

        # Read LDR images
        ldr_images = cv2_read_images(self.file_list[index]["input"])

        # ldr images process
        image_short = ev_alignment(ldr_images[0], floating_exposures[0], 2.24)
        image_medium = ldr_images[1]
        image_long = ev_alignment(ldr_images[2], floating_exposures[2], 2.24)

        image_id = self.file_list[index]["exposure"].split("/")[-1][:4]

        image_short_concat = np.concatenate((ldr_images[0], image_short), 2)
        image_medium_concat = np.concatenate((ldr_images[1], image_medium), 2)
        image_long_concat = np.concatenate((ldr_images[2], image_long), 2)

        img0 = image_short_concat.astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium_concat.astype(np.float32).transpose(2, 0, 1)
        img2 = image_long_concat.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)

        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'image_name': image_id}
        return sample

    def __len__(self):
        return len(self.file_list)