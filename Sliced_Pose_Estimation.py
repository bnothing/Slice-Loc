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
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from datasets import SkyMapLocation
from models import CVM_SkyMap as CVM
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
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-4)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=12)
parser.add_argument('--weight_ori', type=float, help='weight on orientation loss', default=1e1)
parser.add_argument('--weight_infoNCE', type=float, help='weight on infoNCE loss', default=1e4)
parser.add_argument('--rotation_range', type=float, help='range for random orientation', default=180)


args = vars(parser.parse_args())
learning_rate = args['learning_rate']
batch_size = args['batch_size']
weight_ori = args['weight_ori']
training = args['training'] == 'True'
rotation_range = args['rotation_range']
cross_area = args['cross_area'] == 'True'


if cross_area:
    label = 'SkyMap' + '_ori45.0' + '_cross'
else:
    label = 'SkyMap' + '_ori45.0' + '_same'
# label = 'SkyMap_vigor_ori45.0_same'
label = 'SkyMap_ori45.0_same_360'
print('training label:', label)

GrdImg_H = 512  # 256 # original: 375 #224, 256
GrdImg_W = 512  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 375
GrdOriImg_W = 1242
num_thread_workers = 12

SatMap_original_sidelength = 512 
SatMap_process_sidelength = 512

satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
grdimage_transform = transforms.Compose([
        transforms.Resize(size=[GrdImg_H, GrdImg_W]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

ExpRes_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Exp_res/ori_360_prior'
dir_label = ''
if cross_area:
    dir_label += 'cross_'
else:
    dir_label += 'same_'
dir_label += str(rotation_range)

save_dir =ExpRes_dir+'/'+dir_label
city_ = 'Chicago' # Sydney, Chicago

pose_file = 'Pose3DoF'
sloc_file = 'SliceLoc'

res_datas = []

data_root = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
if rotation_range == 45:
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_45'
elif rotation_range == 20:
    pro_dir = '/media/xmt/新加卷1/Data/SkyMap_Process_20'
elif rotation_range == 0:
    pro_dir = '/media/xmt/新加卷1/Data/SkyMap_Process_0'

city_list = None
location_set = SkyMapLocation(root=data_root, pro_dir=pro_dir,
                              rotation_range=rotation_range,
                              sat_resize=SatMap_process_sidelength,
                              cross_area=cross_area,
                              city_list=city_list)
sz1 = int(0.2 * len(location_set))
sz2 = len(location_set) - sz1
location_set_, val_dataset = random_split(location_set, [sz1, sz2])

# location_set_ = location_set
location_loader = DataLoader(location_set, batch_size=1, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
torch.cuda.empty_cache()
CVM_model = CVM(device)
test_model_path = f'models/SkyMap_paral/{label}/11/model.pt'

if not os.path.exists(test_model_path):
    print('not exist modle: ', test_model_path)
else:
    print('load model from: ' + test_model_path)

train_parallar = True # If the model is trained using a single GPU, set it to False.
if train_parallar:
    new_state_dict = OrderedDict()
    state_dict = torch.load(test_model_path)
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

    CVM_model.load_state_dict(new_state_dict, strict=False)
else:
    CVM_model.load_state_dict(torch.load(test_model_path))
CVM_model.to(device)
CVM_model.eval()

err_test = []

t_dict = {}
print('test size:', len(location_set_))
for i, data in tqdm(enumerate(location_loader, 0)):
    sat, grd, city, gt, ori_map, gt_angle, pano_flag = data
    pano_flag = pano_flag[0]
    gt_x, gt_y = gt.numpy().squeeze()
    grd = grd.to(device).squeeze()
    sat = sat.to(device).squeeze()

    with torch.no_grad():
        _, heatmap, ori, _, _, _, _, _, _ = CVM_model(grd, sat)

    # 估计误差角度
    gt_angle = gt_angle.numpy()
    city = city[0]
    cos_gt, sin_gt = np.cos(gt_angle*np.pi/180), np.sin(gt_angle*np.pi/180)

    sW = sat.shape[3]
    heatmap = heatmap.cpu().detach().numpy()
    ori = ori.cpu().detach().numpy()
    prd_locs = []
    slice_oris = []

    for batch_idx in range(sat.shape[0]):
        current_pred = heatmap[batch_idx, :, :, :]
        _, y, x = np.unravel_index(current_pred.argmax(), current_pred.shape)  # 选取热力图中最大值，作为预测的坐标所在

        # 记录角度的预测值
        cos_pred, sin_pred = ori[batch_idx, :, y, x]
        if np.abs(cos_pred) <= 1 and np.abs(sin_pred) <= 1:
            a_acos_pred = math.acos(cos_pred)
            if sin_pred < 0:
                angle_pred = math.degrees(-a_acos_pred) % 360
            else:
                angle_pred = math.degrees(a_acos_pred)
        slice_oris.append(angle_pred)
        prd_locs.append([x, y])
    slice_oris = np.array(slice_oris) # 角度为与正北方向的顺时针夹角值
    pose_mat = np.column_stack([prd_locs, slice_oris])
    pose_dir = osp.join(save_dir, city, pose_file)
    if not osp.exists(pose_dir):
        os.makedirs(pose_dir)
    mat_path = osp.join(pose_dir, pano_flag + '.npy') # x, y, ori
    np.save(mat_path, pose_mat)

    # heatmap_dir = osp.join(save_dir, city, 'heatmap')
    # if not osp.exists(heatmap_dir):
    #     os.makedirs(heatmap_dir)
    # heatmap_path = osp.join(heatmap_dir, pano_flag + '.npy') # x, y, ori
    # np.save(heatmap_path, heatmap)

# for k, v in t_dict.items():
#     print(k, v/len(location_loader))
#     sW = SatMap_process_sidelength
#     slice_locs = []
#     slice_num = sat.shape[0]
#     for batch_idx in range(slice_num):
#         heading = batch_idx*360/slice_num
#         heading += 90 # 转变为与正东方向顺时针夹角值
#         if heading > 360:
#             heading -= 360
#         x, y = prd_locs[batch_idx]
#         x, y = rotate_point(x, y, sW/2, sW/2, -heading)
#         slice_locs.append([x, y])
#     slice_locs = np.array(slice_locs)
#
#     # sloc_err1 在粗差剔除中进行判定的距离
#     temp, inlier_idx, is_available, NFA, sloc_err1 =\
#         loc_util.location_camera_center_NFA2(slice_locs,
#                                              nfa_thr=0.,
#                                              out_nfa=True,
#                                              prd_directs=slice_oris,
#                                              h0=H0_model,
#                                              city=city)
#
#     meter_per_pixel = location_set.meter_per_pixel_dict[city]
#     err2 = np.sqrt((temp[0]-gt_x)**2 + (temp[1]-gt_y)**2) * meter_per_pixel
#     err_test.append(err2)
#     pass
#
# err_test = np.array(err_test)
# print('mean:', np.mean(err_test))