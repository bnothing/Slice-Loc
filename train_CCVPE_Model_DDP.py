import os
import time

os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import argparse
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from datasets import SkyMapDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from losses import infoNCELoss, cross_entropy_loss, orientation_loss
from models import CVM_Slice_Loc as CVM
from tqdm import tqdm
from collections import OrderedDict


torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is: {}".format(device))

parser = argparse.ArgumentParser()
parser.add_argument('--training', choices=('True','False'), default='True')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=2e-4)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=12)
parser.add_argument('--weight_ori', type=float, help='weight on orientation loss', default=1e1)
parser.add_argument('--weight_infoNCE', type=float, help='weight on infoNCE loss', default=1e4)
parser.add_argument('--rotation_range', type=float, help='range for random orientation', default=180)
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--num_workers', type=int, help='number of workers', default=4)
parser.add_argument('--cross_area', choices=('True','False'), default='True')
parser.add_argument('--cross_area', choices=('True','False'), default='True')
parser.add_argument('--data_root', type=str, default=r'/mnt/Disk_2/xiongmingtao/Datas/SkyMap',
                    help='path to the root folder of all dataset')
parser.add_argument('--pro_root', type=str, default=r'/mnt/Disk_2/xiongmingtao/Datas/SkyMap_Processing_45',
                    help='path to the root folder of all processed sliced image')


args = vars(parser.parse_args())
learning_rate = args['learning_rate']
batch_size = args['batch_size']
weight_ori = args['weight_ori']
weight_infoNCE = args['weight_infoNCE']
training = args['training'] == 'True'
rotation_range = args['rotation_range']
local_rank = args['local_rank']
num_workers = args['num_workers']
cross_area = args['cross_area'] == 'True'
data_root = args['data_root']
pro_root = args['pro_root']

if cross_area:
    label = 'DReSS-D' + '_ori45.0' + '_cross'
else:
    label = 'DReSS-D' + '_ori45.0' + '_same'
print('training label:', label)

# 多线程的初始化
if local_rank != -1:
    torch.cuda.set_device(local_rank)
    device=torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

train_set = SkyMapDataset(root=data_root, pro_dir=pro_root,
                          rotation_range=rotation_range,
                          sat_resize=512,
                          cross_area=cross_area,
                          train=training)
train_sampler = DistributedSampler(train_set, shuffle=True)
train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True,
                              num_workers=num_workers, drop_last=False,sampler=train_sampler)
if local_rank == 0:
    print('total image number:', len(train_set))
    print('total data size:', len(train_loader))

torch.cuda.empty_cache()
CVM_model = CVM(device)

if training:
    print("start trainning:", local_rank)
    print("data size: ", len(train_loader))
    torch.cuda.empty_cache()
    CVM_model.to(device)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        CVM_model = nn.parallel.DistributedDataParallel(CVM_model, device_ids=[local_rank],
                                                    output_device=local_rank,find_unused_parameters=True)

    for param in CVM_model.parameters():
        param.requires_grad = True

    params = [p for p in CVM_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))

    global_step = 0
    # with torch.autograd.set_detect_anomaly(True):

    rank = dist.get_rank()
    for epoch in range(0, 13):  # loop over the dataset multiple times
        train_sampler.set_epoch(epoch)  # shuffle
        running_loss = 0.0
        CVM_model.train()
        for i, data in tqdm(enumerate(train_loader, 0)):
            sat, grd, gt, gt_with_ori, gt_orientation, orientation_angle = [item.to(device) for item in data]

            gt_flattened = torch.flatten(gt, start_dim=1)
            gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

            gt_bottleneck = nn.MaxPool2d(64, stride=64)(gt_with_ori).contiguous()  # [B, 16, 8, 8]
            gt_bottleneck2 = nn.MaxPool2d(32, stride=32)(gt_with_ori).contiguous()
            gt_bottleneck3 = nn.MaxPool2d(16, stride=16)(gt_with_ori).contiguous()
            gt_bottleneck4 = nn.MaxPool2d(8, stride=8)(gt_with_ori).contiguous()
            gt_bottleneck5 = nn.MaxPool2d(4, stride=4)(gt_with_ori).contiguous()
            gt_bottleneck6 = nn.MaxPool2d(2, stride=2)(gt_with_ori).contiguous()  # [B, 16, 256, 256]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd, sat)

            loss_ori = torch.sum(torch.sum(torch.square(gt_orientation-ori), dim=1, keepdim=True) * gt) / logits_flattened.size()[0]        
            loss_infoNCE = infoNCELoss(torch.flatten(matching_score_stacked, start_dim=1), torch.flatten(gt_bottleneck, start_dim=1))
            loss_infoNCE2 = infoNCELoss(torch.flatten(matching_score_stacked2, start_dim=1), torch.flatten(gt_bottleneck2, start_dim=1))
            loss_infoNCE3 = infoNCELoss(torch.flatten(matching_score_stacked3, start_dim=1), torch.flatten(gt_bottleneck3, start_dim=1))
            loss_infoNCE4 = infoNCELoss(torch.flatten(matching_score_stacked4, start_dim=1), torch.flatten(gt_bottleneck4, start_dim=1))
            loss_infoNCE5 = infoNCELoss(torch.flatten(matching_score_stacked5, start_dim=1), torch.flatten(gt_bottleneck5, start_dim=1))
            loss_infoNCE6 = infoNCELoss(torch.flatten(matching_score_stacked6, start_dim=1), torch.flatten(gt_bottleneck6, start_dim=1))
            loss_ce = cross_entropy_loss(logits_flattened, gt_flattened)
            loss = loss_ce + weight_infoNCE*(loss_infoNCE+loss_infoNCE2+loss_infoNCE3+loss_infoNCE4+loss_infoNCE5+loss_infoNCE6)/6 + weight_ori*loss_ori

            global_step += 1
            loss.backward()
            if torch.any(torch.isnan(loss)):
                print("Loss contains NaN, idx is", i)
            else:
                optimizer.step()
                running_loss += loss.item()

            if i % 200 == 199 and local_rank == 0:    # print every 200 mini-batches
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        if epoch % 3 == 2: # 每3个epoch调整一次学习率
            learning_rate *= 0.75
            rank = dist.get_rank()
            if rank == 0:
                print('learning rate:', learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        if rank == 0:
            model_dir = 'models/SkyMap_paral/'+label+'/' + str(epoch) + '/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(CVM_model.state_dict(), model_dir+'model.pt') # saving model

    print('Finished Training')
