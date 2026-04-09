import os
import os.path as osp
import time

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import sys
import argparse
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from datasets import ReliableLoc
from models import CVM_Slice_Loc as CVM
from processing import location_camera as loc_util
from tqdm import tqdm
from collections import OrderedDict

torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is: {}".format(device))

parser = argparse.ArgumentParser()
parser.add_argument('--training', choices=('True','False'), default='True')
parser.add_argument('--cross_area', choices=('True','False'), default='True')
parser.add_argument('--rotation_range', type=float, help='range for random orientation', default=180)
parser.add_argument('--save_root', type=str, help='path for saving results')
parser.add_argument('--data_root', type=str, default=r'/mnt/Disk_2/xiongmingtao/Datas/SkyMap',
                    help='path to the root folder of all dataset')
parser.add_argument('--pro_root', type=str, default=r'/mnt/Disk_2/xiongmingtao/Datas/SkyMap_Processing_45',
                    help='path to the root folder of all processed sliced image')

args = vars(parser.parse_args())
rotation_range = args['rotation_range']
cross_area = args['cross_area'] == 'True'
ExpRes_dir = args['save_root']
data_root = args['data_root']
pro_root = args['pro_root']

nfa_thr = 0.
print('use NFA threshold:', nfa_thr)

dir_label = ''
if cross_area:
    dir_label += 'cross_'
else:
    dir_label += 'same_'
dir_label += str(rotation_range)
save_dir = osp.join(ExpRes_dir, dir_label)

city_list = None
SliceLoc = ReliableLoc(root=data_root, pro_dir=pro_root,
                              nfa_thr=nfa_thr,
                              pose_dir=save_dir,
                              rotation_range=rotation_range,
                              sat_resize=512,
                              cross_area=cross_area,
                              city_list=city_list)
print('Estimating camera 3DoF pose:')
SliceLoc.processing()
print('Write final result')
SliceLoc.save_results()
