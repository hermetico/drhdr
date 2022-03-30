import os
import cv2
import shutil
import random
from tqdm import tqdm
from pathlib import Path
from tqdm.contrib.concurrent import thread_map

SRC = Path("/mnt/hdd/shared/datasets/")
ROOT = SRC / "ntire-hdr-2022"
DEST = SRC / "ntire-hdr-2022-clean-256"

GT = "_gt.png"
SHORT = "_short.png"
MEDIUM = "_medium.png"
LONG = "_long.png"
ALIGN ="_alignratio.npy"
EXPOSURE ="_exposures.npy"
KEY = LONG
SKIP = 0
VALID_FILES = 250 # Images for validation
WORKERS = 8

D = 256 # Patch size
S = 256 # Stride



def split_training_sample(params):
    i, p = params
    mode = "train"
    c_folder = ""
    c_folder = p.parts[-2] +  "/"

    desc = p.name.replace(KEY, "")
    #tqbar.set_description(desc, refresh=True)
    gt_p = p.as_posix()
    
    align_p = gt_p.replace(KEY, ALIGN)
    exposure_p = gt_p.replace(KEY, EXPOSURE)
    
    base_folder = Path(make_path(gt_p, DEST / mode, c_folder))
    create_path(base_folder)
    # copy numpy files
    shutil.copy(align_p, (base_folder / ALIGN.lstrip("_")).as_posix())
    shutil.copy(exposure_p, (base_folder / EXPOSURE.lstrip("_")).as_posix())
    # create patches
    for _type in [SHORT, MEDIUM, LONG, GT]:
        cp = gt_p.replace(KEY, _type)
        img = cv2.imread(cp, cv2.IMREAD_UNCHANGED)
        W, H = img.shape[:2]
        w, h = 0, 0
        subindex = 0
        while w + D < W:
            h = 0
            while h + D < H:
                
                # create the new folder structure
                # BASE/img-index/patch-subindex_type.png
                patch = img[w:w+D, h:h+D, ...]
                new_file = base_folder /  (f"000{subindex}"[-3:] + _type)
                cv2.imwrite(new_file.as_posix(), patch)
                h += S
                subindex += 1
            w += S

def split_validation_sample(params):
    i, p = params
    mode = "valid"
    c_folder = ""
    c_folder = p.parts[-2] +  "/"

    desc = p.name.replace(KEY, "")
    #tqbar.set_description(desc, refresh=True)
    gt_p = p.as_posix()
    
    align_p = gt_p.replace(KEY, ALIGN)
    exposure_p = gt_p.replace(KEY, EXPOSURE)
    
    base_folder = Path(make_path(gt_p, DEST / mode, c_folder))
    create_path(base_folder)
    # copy numpy files
    shutil.copy(align_p, (base_folder / ALIGN.lstrip("_")).as_posix())
    shutil.copy(exposure_p, (base_folder / EXPOSURE.lstrip("_")).as_posix())

    # create patches
    for _type in [SHORT, MEDIUM, LONG, GT]:
        cp = gt_p.replace(KEY, _type)
        shutil.copy(cp, (base_folder / _type.lstrip("_")).as_posix())


def sort_key(filepath):
    # sorts by id
    return  int(filepath.name.replace(KEY, ""))

def make_path(gt_path, dest_path, sub_folder):
    new_path = gt_path.replace(KEY, "")
    new_path = new_path.replace(sub_folder, "")
    new_path = new_path.replace(ROOT.as_posix(), dest_path.as_posix())
    return new_path

def create_path(path):
    os.makedirs(path.as_posix(), exist_ok=True)

if __name__ == "__main__":

    pattern = "*Train*/*" + KEY
    paths = list(sorted(ROOT.glob(pattern), key=sort_key))
    
    random.shuffle(paths)
    
    print("Training data")
    training_paths = paths[:-VALID_FILES]
    thread_map(split_training_sample, list(enumerate(training_paths)), max_workers=WORKERS)
    
    print("Validation data")
    validation_paths = paths[-VALID_FILES:]
    thread_map(split_validation_sample, list(enumerate(validation_paths)), max_workers=WORKERS)




