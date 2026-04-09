import os
import os.path as osp
import time

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import sys
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from datasets import SkyMapLocation
from models import CVM_Slice_Loc as CVM
from processing import location_camera as loc_util
from tqdm import tqdm
from collections import OrderedDict

def loc_heatmap(A):
    B, H, W = A.shape
    # 1. 先展平成 [B, H*W]，在 dim=1 上取最大值索引
    flat_idx = A.view(B, -1).argmax(dim=1)  # shape [B]
    # 2. 还原行列坐标
    rows = flat_idx // W  # 行号 tensor [B]
    cols = flat_idx % W  # 列号 tensor [B]
    # 3. 拼成 [B, 2]
    coords = torch.stack([cols, rows], dim=1)  # shape [B, 2]
    return coords


torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is: {}".format(device))

parser = argparse.ArgumentParser()
parser.add_argument('--training', choices=('True','False'), default='True')
parser.add_argument('--cross_area', choices=('True','False'), default='True')
parser.add_argument('--rotation_range', type=float, help='range for random orientation', default=45)
parser.add_argument('--ckpt_path', type=str, help='path for checkpoints')
parser.add_argument('--save_root', type=str, help='path for saving results')
parser.add_argument('--data_root', type=str, default=r'/mnt/Disk_2/xiongmingtao/Datas/SkyMap',
                    help='path to the root folder of all dataset')
parser.add_argument('--pro_root', type=str, default=r'/mnt/Disk_2/xiongmingtao/Datas/SkyMap_Processing_45',
                    help='path to the root folder of all processed sliced image')


args = vars(parser.parse_args())
rotation_range = args['rotation_range']
cross_area = args['cross_area'] == 'True'
ckpt_path = args['ckpt_path']
ExpRes_dir = args['save_root']
data_root = args['data_root']
pro_root = args['pro_root']

dir_label = ''
if cross_area:
    dir_label += 'cross_'
else:
    dir_label += 'same_'
dir_label += str(rotation_range)

save_dir =ExpRes_dir+'/'+dir_label

pose_file = 'Pose3DoF'
sloc_file = 'SliceLoc'

res_datas = []

city_list = None
num_thread_workers = 12
location_set = SkyMapLocation(root=data_root, pro_dir=pro_root,
                              rotation_range=rotation_range,
                              sat_resize=512,
                              cross_area=cross_area,
                              city_list=city_list)
location_loader = DataLoader(location_set, batch_size=1, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
torch.cuda.empty_cache()
CVM_model = CVM(device)

if not os.path.exists(ckpt_path):
    print('not exist modle: ', ckpt_path)
else:
    print('load model from: ', ckpt_path)

train_parallar = True # If the model is trained using a single GPU, set it to False.
if train_parallar:
    new_state_dict = OrderedDict()
    state_dict = torch.load(ckpt_path)
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

    CVM_model.load_state_dict(new_state_dict, strict=False)
else:
    CVM_model.load_state_dict(torch.load(ckpt_path))
CVM_model.to(device)
CVM_model.eval()

err_test = []

t_dict = {}
print('test size:', len(location_set))
for i, data in tqdm(enumerate(location_loader, 0)):
    sat, grd, city, gt, ori_map, gt_angle, pano_flag = data
    pano_flag = pano_flag[0]
    city = city[0]
    gt_x, gt_y = gt.numpy().squeeze()
    grd = grd.to(device).squeeze()
    sat = sat.to(device).squeeze()

    with torch.no_grad():
        _, heatmap, ori, _, _, _, _, _, _ = CVM_model(grd, sat)

    pd_loc = loc_heatmap(heatmap.squeeze(1))
    pd_ori = ori[torch.arange(pd_loc.size(0)), :, pd_loc[:,1], pd_loc[:,0]]
    pose_mat = np.column_stack([pd_loc.cpu().numpy(), pd_ori.cpu().numpy()])
    pose_dir = osp.join(save_dir, city, pose_file)
    if not osp.exists(pose_dir):
        os.makedirs(pose_dir)
    mat_path = osp.join(pose_dir, pano_flag + '.npy') # x, y, ori
    np.save(mat_path, pose_mat)