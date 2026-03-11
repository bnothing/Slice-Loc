import os
import os.path as osp
from torch.utils.data import Dataset
import PIL.Image
import PIL.ImageDraw
import torch
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
import math
import cv2
from processing.preprocessing import GeoTrans, decom_pano_name, decom_pano_name_vigor, statistic_data_pixel
from torchvision import transforms

import pandas as pd
from scipy.stats import norm
from processing import location_camera as loc_util
from processing import Mercator as geo_util
import json
from multiprocessing import Pool
from tqdm import tqdm

torch.manual_seed(17)
np.random.seed(0)

GrdImg_H = 512  # 256 # original: 375 #224, 256
GrdImg_W = 512  # 1024 # original:1242 #1248, 1024
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

# ---------------------------------------------------------------------------------
# SkyMap
def compute_meters_per_pixel(sat_folder, crop_size, resize):
    # 计算卫星图像分辨率, 每个城市分辨率不同
    # 卫星影像先进行裁剪，再进行resize()
    files = [f for f in os.listdir(sat_folder)]
    file = files[0]
    sat = PIL.Image.open(osp.join(sat_folder, file, 'satellite.jpg'), 'r')
    tl_path = osp.join(sat_folder, file, 'tl_pos.txt')
    i_trans = GeoTrans(json_path=None, tl_path=tl_path, sat_size=sat.width)
    lon_sol = i_trans.lon_ratio * abs(i_trans.sol_lon)
    lat_sol = i_trans.lat_ratio*abs(i_trans.sol_lat)
    sat_sol = (lon_sol+lat_sol)/2
    sat_sol *= (crop_size/resize)
    return sat_sol

# theta 为正逆时针旋转，为负顺时针旋转
def rotate_point(x, y, x_c, y_c, theta):
    # 旋转矩阵
    theta = theta * (np.pi/180)
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],
                                [-np.sin(theta), np.cos(theta)]])
    # 计算相对中心点的坐标
    relative_coords = np.array([x - x_c, y - y_c])
    # 应用旋转矩阵
    rotated_coords = rotation_matrix.dot(relative_coords)
    # 计算旋转后的坐标
    x_new = rotated_coords[0] + x_c
    y_new = rotated_coords[1] + y_c
    return x_new, y_new

# 用于txt文件中加载数据列表的类，
# 加载train.txt, test.txt文件
# 用与SkyMap定位方法
class SkyMap_Loader():
    def __init__(self, pro_dir, training, cross_area, slice_txt=False):
        self.pro_dir = pro_dir
        self.training = training
        self.cross_area = cross_area
        self.slice_txt = slice_txt

    def get_citys(self):
        if self.cross_area:
            if self.training:
                # 训练时使用
                city_list = ['Tokyo', 'Rio', 'London']
            else:
                # 测试时使用另外的城市
                city_list = ['Chicago', 'Johannesburg', 'Sydney']
        else:
            city_list = ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
        return city_list

    # 根据输入加载数据列表
    def run(self, city):
        if self.training:
            if self.slice_txt: # Slice CVGL
                if self.cross_area:
                    txt_name = 'slice_cross.txt'
                else:
                    txt_name = 'slice_same_train.txt'
            else:
                if self.cross_area:
                    txt_name = 'pano_label_balanced.txt'
                else:
                    txt_name = 'same_area_balanced_train.txt'
        else:
            if self.slice_txt: # Slice CVGL
                if self.cross_area:
                    txt_name = 'slice_cross.txt'
                else:
                    txt_name = 'slice_same_test.txt'
            else:
                if self.cross_area:
                    txt_name = 'pano_label_balanced.txt'
                else:
                    txt_name = 'same_area_balanced_test.txt'
                    # txt_name = 'same_area_balanced_train.txt' # 测试模型的过拟合
                    # txt_name = 'pano_label_balanced.txt'

        file_path = osp.join(self.pro_dir, city, txt_name)
        with open(file_path, 'r') as f:
            grd_datas = f.readlines()
        return grd_datas

# 加载skymap数据，在原有的基础上修改，误差已生成
# 角度误差已添加在全景影像的slice中，偏移误差从train.txt中获取
class SkyMapDataset(Dataset):
    def __init__(self, root, pro_dir,
                 transform=None, rotation_range=10, sat_resize=512, cross_area=True, train=True, city_list=None):
        self.root = root
        self.pro_root = pro_dir
        self.train = train

        self.shift_range_pixel = 160 # in terms of pixels
        self.rotation_range = rotation_range  # in terms of degree
        self.slice_num = 12

        self.SatMap_process_sidelength = 640 # 添加误差后中心裁切的图像的大小
        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]
        else:
            self.satmap_transform = satmap_transform
            self.grdimage_transform = grdimage_transform

        self.slice_flag = 'slice'
        self.sat_dir = 'sat_img'

        self.meter_per_pixel_dict = {}
        self.grd_datas = [] # path, o_rand, x_rand, y_rand, x_gt, y_gt
        skymap_loder = SkyMap_Loader(pro_dir, self.train, cross_area, True)
        if city_list is not None:
            self.city_list = city_list
        else:
            self.city_list = skymap_loder.get_citys()
        print('load citys:', self.city_list)
        for city in self.city_list:
            grd_datas = skymap_loder.run(city)
            self.grd_datas.extend([osp.join(city, self.slice_flag, data[:-1]) for data in grd_datas])
            self.meter_per_pixel_dict[city] = compute_meters_per_pixel(
                osp.join(self.root, city, self.sat_dir),
                self.SatMap_process_sidelength,
                sat_resize
            )


    def __len__(self):
        return len(self.grd_datas)

    def get_file_list(self):
        return self.grd_data

    def __getitem__(self, idx):
        # 根据编号读取数据并返回监督所需的真值
        # 先添加误差再进行定位数据的初始方向对齐
        grd_data = self.grd_datas[idx]
        parts = grd_data.split()
        i_path, o_rand, x_rand, y_rand, x_gt, y_gt = parts
        # o_rand: 即在切片时，顺时针方向偏离一定的度数
        o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)

        grd_path = osp.join(self.pro_root, i_path)
        temp = i_path.split('/')
        city, _, flag, i_name = temp
        sat_path = osp.join(self.root, city, self.sat_dir, flag, 'satellite.jpg')

        ox, oy = float(x_gt), float(y_gt)
        heading = float(osp.splitext(i_name)[0]) * 360 / self.slice_num
        heading += 90 # 转变为与正东方向顺时针夹角值
        if heading > 360:
            heading -= 360

        # =================== read satellite map ===================================
        with PIL.Image.open(sat_path, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
        cx, cy = SatMap.width/2, SatMap.height/2

        sat0 = np.array(sat_map)
        sat0 = cv2.circle(sat0, center=(int(ox), int(oy)), radius=3, thickness=1,
                             color=(0, 0, 255))
        sat0 = cv2.circle(sat0, center=(int(cx), int(cy)), radius=3, thickness=1,
                             color=(255, 0, 0))

        # randomly generate shift
        shift_range = self.shift_range_pixel
        gt_shift_x = x_rand * shift_range
        gt_shift_y = y_rand * shift_range

        sat_rand_shift = \
            sat_map.transform(
                sat_map.size, PIL.Image.AFFINE,
                (1, 0, -gt_shift_x,
                 0, 1, -gt_shift_y),
                resample=PIL.Image.BILINEAR)
        # randomly generate roation
        random_ori = o_rand * self.rotation_range  # 0 means the arrow in aerial image heading Easting, counter-clockwise increasing

        ox += gt_shift_x
        oy += gt_shift_y
        sat2 = np.array(sat_rand_shift)
        sat2 = cv2.circle(sat2, center=(int(ox), int(oy)), radius=3, thickness=1,
                             color=(0, 0, 255))
        cx += gt_shift_x
        cy += gt_shift_y
        sat2 = cv2.circle(sat2, center=(int(cx), int(cy)), radius=3, thickness=1,
                             color=(255, 0, 0))

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        with PIL.Image.open(grd_path, 'r') as GrdImg:
            grd_img_left = GrdImg.convert('RGB')
            grd_test = np.array(grd_img_left)
            if self.grdimage_transform is not None:
                grd_img_left = self.grdimage_transform(grd_img_left)

        grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

        sat_align_cam = sat_rand_shift.rotate(heading)  # make the east direction the vehicle heading
        sat3 = np.array(sat_align_cam)
        ox, oy = rotate_point(ox, oy, SatMap.width/2, SatMap.height/2, heading)
        cx, cy = rotate_point(cx, cy, SatMap.width/2, SatMap.height/2, heading)

        sat3 = cv2.circle(sat3, center=(int(ox), int(oy)), radius=3, thickness=1,
                             color=(0, 0, 255))
        sat3 = cv2.circle(sat3, center=(int(cx), int(cy)), radius=3, thickness=1,
                             color=(255, 0, 0))
        dx = ox - SatMap.width/2
        dy = oy - SatMap.height/2
        sat_map = TF.center_crop(sat_align_cam, self.SatMap_process_sidelength)

        sat4 = np.array(sat_map.resize((512, 512)))

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # gt heat map
        x_offset = int(-dx * sat_map.shape[1]/self.SatMap_process_sidelength)
        y_offset = int(-dy * sat_map.shape[1] / self.SatMap_process_sidelength)

        x, y = np.meshgrid(np.linspace(-256 + x_offset, 256 + x_offset, 512),
                           np.linspace(-256 + y_offset, 256 + y_offset, 512))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        gt = torch.tensor(gt)  # 生成二维高斯分布，以车辆所在位置为中心
        if self.train:
            # orientation gt
            orientation_angle = 90 + random_ori
            if orientation_angle < 0:
                orientation_angle = orientation_angle + 360
            elif orientation_angle > 360:
                orientation_angle = orientation_angle - 360

            gt_with_ori = np.zeros([16, 512, 512], dtype=np.float32)
            index = int(orientation_angle // 22.5)
            ratio = (orientation_angle % 22.5) / 22.5
            if index == 0:
                gt_with_ori[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * (1 - ratio)
                gt_with_ori[15, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * ratio
            else:
                gt_with_ori[16 - index, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * (1 - ratio)
                gt_with_ori[16 - index - 1, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * ratio
            gt_with_ori = torch.tensor(gt_with_ori)

            orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi / 180))
            orientation_map[1, :, :] = np.sin(orientation_angle * np.pi / 180)


            gt0 = gt.numpy().squeeze()
            gty, gtx = np.unravel_index(gt0.argmax(), gt0.shape)
            sat4 = cv2.circle(sat4, center=(int(gtx), int(gty)), radius=3, thickness=1,
                                 color=(0, 0, 255))

            # 返回值，主要返回heatmap: 位移与旋转角
            # gt: 位移在sat_map各像素处旋转角度的真值
            # gt_with_ori: 在sat_map各像素处旋转角度的真值
            # orientation_map： 旋转角的cos与sin值
            # orientation_angle：旋转角真值，度, 认为地面初始都是朝北看，旋转该角度后即可与卫星重合
            return sat_map, grd_left_imgs[0], gt, gt_with_ori, orientation_map, orientation_angle
        else:
            return sat_map, grd_left_imgs[0], gt, city


# 使用相机的拍摄位置作为slice定位的真值
class SkyMapDataset_cemera(Dataset):
    def __init__(self, root, pro_dir,
                 transform=None, rotation_range=10, sat_resize=512, cross_area=True, train=True, city_list=None):
        self.root = root
        self.pro_root = pro_dir
        self.train = train

        self.shift_range_pixel = 160 # in terms of pixels
        self.rotation_range = rotation_range  # in terms of degree

        self.SatMap_process_sidelength = 640 # 添加误差后中心裁切的图像的大小
        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.slice_flag = 'slice'
        self.sat_dir = 'sat_img'

        self.meter_per_pixel_dict = {}
        self.grd_datas = [] # path, o_rand, x_rand, y_rand, x_gt, y_gt
        skymap_loder = SkyMap_Loader(pro_dir, self.train, cross_area, True)
        if city_list is not None:
            self.city_list = city_list
        else:
            self.city_list = skymap_loder.get_citys()
        print('load citys:', self.city_list)
        for city in self.city_list:
            grd_datas = skymap_loder.run(city)
            self.grd_datas.extend([osp.join(city, self.slice_flag, data[:-1]) for data in grd_datas])
            self.meter_per_pixel_dict[city] = compute_meters_per_pixel(
                osp.join(self.root, city, self.sat_dir),
                self.SatMap_process_sidelength,
                sat_resize
            )


    def __len__(self):
        return len(self.grd_datas)

    def get_file_list(self):
        return self.grd_data

    def __getitem__(self, idx):
        # 根据编号读取数据并返回监督所需的真值
        # 先添加误差再进行定位数据的初始方向对齐
        grd_data = self.grd_datas[idx]
        parts = grd_data.split()
        i_path, o_rand, x_rand, y_rand, x_gt, y_gt = parts
        # o_rand: 即在切片时，顺时针方向偏离一定的度数
        o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)

        grd_path = osp.join(self.pro_root, i_path)
        temp = i_path.split('/')
        city, _, flag, i_name = temp
        sat_path = osp.join(self.root, city, self.sat_dir, flag, 'satellite.jpg')

        ox, oy = float(x_gt), float(y_gt)
        heading = float(osp.splitext(i_name)[0]) * 360 / 12
        heading += 90 # 转变为与正东方向顺时针夹角值
        if heading > 360:
            heading -= 360

        # =================== read satellite map ===================================
        with PIL.Image.open(sat_path, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
        cx, cy = SatMap.width/2, SatMap.height/2
        ox, oy = cx, cy

        # sat0 = np.array(sat_map)
        # sat0 = cv2.circle(sat0, center=(int(ox), int(oy)), radius=3, thickness=1,
        #                      color=(0, 0, 255))
        # sat0 = cv2.circle(sat0, center=(int(cx), int(cy)), radius=3, thickness=1,
        #                      color=(255, 0, 0))

        # randomly generate shift
        shift_range = self.shift_range_pixel
        gt_shift_x = x_rand * shift_range
        gt_shift_y = y_rand * shift_range

        sat_rand_shift = \
            sat_map.transform(
                sat_map.size, PIL.Image.AFFINE,
                (1, 0, -gt_shift_x,
                 0, 1, -gt_shift_y),
                resample=PIL.Image.BILINEAR)
        # randomly generate roation
        random_ori = o_rand * self.rotation_range  # 0 means the arrow in aerial image heading Easting, counter-clockwise increasing

        ox += gt_shift_x
        oy += gt_shift_y
        # sat2 = np.array(sat_rand_shift)
        # sat2 = cv2.circle(sat2, center=(int(ox), int(oy)), radius=3, thickness=1,
        #                      color=(0, 0, 255))
        cx += gt_shift_x
        cy += gt_shift_y
        # sat2 = cv2.circle(sat2, center=(int(cx), int(cy)), radius=3, thickness=1,
        #                      color=(255, 0, 0))

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        with PIL.Image.open(grd_path, 'r') as GrdImg:
            grd_img_left = GrdImg.convert('RGB')
            grd_test = np.array(grd_img_left)
            if self.grdimage_transform is not None:
                grd_img_left = self.grdimage_transform(grd_img_left)

        grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

        sat_align_cam = sat_rand_shift.rotate(heading)  # make the east direction the vehicle heading
        sat3 = np.array(sat_align_cam)
        ox, oy = rotate_point(ox, oy, SatMap.width/2, SatMap.height/2, heading)
        cx, cy = rotate_point(cx, cy, SatMap.width/2, SatMap.height/2, heading)

        sat3 = cv2.circle(sat3, center=(int(ox), int(oy)), radius=3, thickness=1,
                             color=(0, 0, 255))
        sat3 = cv2.circle(sat3, center=(int(cx), int(cy)), radius=3, thickness=1,
                             color=(255, 0, 0))
        dx = ox - SatMap.width/2
        dy = oy - SatMap.height/2
        sat_map = TF.center_crop(sat_align_cam, self.SatMap_process_sidelength)

        sat4 = np.array(sat_map.resize((512, 512)))

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # gt heat map
        x_offset = int(-dx * sat_map.shape[1]/self.SatMap_process_sidelength)
        y_offset = int(-dy * sat_map.shape[1] / self.SatMap_process_sidelength)

        x, y = np.meshgrid(np.linspace(-256 + x_offset, 256 + x_offset, 512),
                           np.linspace(-256 + y_offset, 256 + y_offset, 512))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        gt = torch.tensor(gt)  # 生成二维高斯分布，以车辆所在位置为中心

        # orientation gt
        orientation_angle = 90 + random_ori
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360

        gt_with_ori = np.zeros([16, 512, 512], dtype=np.float32)
        index = int(orientation_angle // 22.5)
        ratio = (orientation_angle % 22.5) / 22.5
        if index == 0:
            gt_with_ori[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * (1 - ratio)
            gt_with_ori[15, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * ratio
        else:
            gt_with_ori[16 - index, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * (1 - ratio)
            gt_with_ori[16 - index - 1, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * ratio
        gt_with_ori = torch.tensor(gt_with_ori)

        orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi / 180))
        orientation_map[1, :, :] = np.sin(orientation_angle * np.pi / 180)


        # gt0 = gt.numpy().squeeze()
        # gty, gtx = np.unravel_index(gt0.argmax(), gt0.shape)
        # sat4 = cv2.circle(sat4, center=(int(gtx), int(gty)), radius=3, thickness=1,
        #                      color=(0, 0, 255))

        # 返回值，主要返回heatmap: 位移与旋转角
        # gt: 位移在sat_map各像素处旋转角度的真值
        # gt_with_ori: 在sat_map各像素处旋转角度的真值
        # orientation_map： 旋转角的cos与sin值
        # orientation_angle：旋转角真值，度, 认为地面初始都是朝北看，旋转该角度后即可与卫星重合
        if self.train:
            return sat_map, grd_left_imgs[0], gt, gt_with_ori, orientation_map, orientation_angle
        else:
            return sat_map, grd_left_imgs[0], gt, gt_with_ori, orientation_map, orientation_angle, city


city_dict = {
    'CHG':'Chicago',
    'SD':'Sydney',
    'LD':'London',
    'TP':'Taipei',
    'JNB':'Johannesburg',
    'RIO':'Rio',
    'TK':'Tokyo'
}

class DictToAttributes:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# 使用均匀分布+正态分布模型
class NullHypothesis:
    def  __init__(self, pro_dir):
        self.pro_dir = pro_dir
        self.param_txt = 'norm_param.txt'
        self.param_txt3 = 'pdf_theta.txt'
        self.param_dict = self.load_param()
        self.param_dict2 = self.load_param2()
        # self.param_dict3 = self.load_param3()

    # 使用概率密度列表计算概率
    def probability3(self, x):
        pdf, step = self.param_dict3.pdf, self.param_dict3.step
        ind = int(round(x // step))
        d = x % step
        if x >= 180.:
            return 1
        if ind > 0:
            p0 = pdf[ind-1]
            y = pdf[ind] - pdf[ind-1]
        else:
            p0 = 0
            y = pdf[ind]
        return p0 + y*d

    # 加载概率密度列表
    def load_param3(self):
        txt_path = osp.join(self.pro_dir, self.param_txt3)
        pdf_mat = np.loadtxt(txt_path)
        step = 180 / len(pdf_mat)
        temp_dict = {
            'pdf':pdf_mat,
            'step':step

        }
        return DictToAttributes(**temp_dict)

    # 使用两段直线的概率参数
    def load_param2(self):
        x1, y1 = 49.82, 5.48496e-3
        x2, y2 = 126.474, 3.59354e-4
        k = (y1-y2)/(x1-x2)
        b = y1-k*x1
        x0 = -b/k
        int0 = x1*y1 + (x0-x1)*y1/2
        temp_dict = {
            'x1':x1,
            'y1':y1,
            'k':k,
            'b':b,
            'x0':x0,
            'int0':int0
        }
        return DictToAttributes(**temp_dict)

    # 计算两段直线的概率分布函数
    def probability2(self, x):
        p = self.param_dict2
        x1, y1, k, b, x0, int0 = \
            p.x1, p.y1, p.k, p.b, p.x0, p.int0
        if x < x1:
            pt = x*y1/int0
            return x*y1/int0
        elif x < x0:
            pt = (x1*y1 + (y1+k*x+b)*(x-x1)/2)/int0
            return (x1*y1 + (y1+k*x+b)*(x-x1)/2)/int0
        else:
            return 1

    def load_param(self):
        txt_path = osp.join(self.pro_dir, self.param_txt)
        data_mat = np.loadtxt(txt_path)
        mu, sigma, xr, xl = data_mat
        pdf0 = norm.pdf(xr, mu, sigma)
        cdf0 = norm.cdf(xr, mu, sigma)
        a0 = pdf0*xr+1-cdf0
        temp_dict = {
            'mu':mu,
            'sigma':sigma,
            'xr':xr,
            'xl':xl,
            'pdf0':pdf0,
            'cdf0':cdf0,
            'a0':a0
        }
        return DictToAttributes(**temp_dict)

    def probability(self, x):
        p = self.param_dict
        mu, sig, xr, pdf0, cdf0, a0 = \
            p.mu, p.sigma, p.xr, p.pdf0, p.cdf0, p.a0

        if x < xr:
            return x*pdf0/a0
        else:
            return (xr*pdf0+norm.cdf(x, mu, sig)-cdf0)/a0

# 用于定位一整张图像
class SkyMapLocation(Dataset):
    def __init__(self, root, pro_dir,
                 transform=None, rotation_range=10, sat_resize=512, cross_area=True, city_list=None):
        self.root = root
        self.pro_root = pro_dir

        self.rotation_range = rotation_range  # in terms of degree
        self.shift_range_pixel = 160 # in terms of pixels

        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'
        self.slice_flag = 'slice'
        self.sat_dir = 'sat_img'

        self.slice_num = 12 # 实际使用的切片图像个数
        self.direct_num = 12 # 总的切片图像个数
        self.SatMap_process_sidelength = 640 # 添加误差后中心裁切的图像的大小
        self.meter_per_pixel_dict = {}

        self.image_list = []
        self.citys = []
        skymap_loder = SkyMap_Loader(pro_dir, False, cross_area, False)
        if city_list is not None:
            self.city_list = city_list
        else:
            self.city_list = skymap_loder.get_citys()
        print('load citys:', self.city_list)
        for city in self.city_list:
            grd_datas = skymap_loder.run(city)
            self.image_list.extend([data[:-1] for data in grd_datas])
            self.citys.extend([city]*len(grd_datas))
            self.meter_per_pixel_dict[city] = compute_meters_per_pixel(
                osp.join(self.root, city, self.sat_dir),
                self.SatMap_process_sidelength,
                sat_resize
            )
        self.tata = 1


    def __len__(self):
        return len(self.image_list)

    def get_file_list(self):
        return self.image_list

    def __getitem__(self, idx):
        # 根据编号读取数据并构建模型的输入数据：grd, sat
        grd_data = self.image_list[idx]
        grd_name, o_rand, x_rand, y_rand = grd_data.split()
        o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
        info = decom_pano_name(grd_name)
        pano_flag = info['flag']
        city = self.citys[idx]

        # 读取卫星数据并处理
        sat_path = osp.join(self.root, city, self.sat_dir, pano_flag, 'satellite.jpg')
        with PIL.Image.open(sat_path, 'r') as SatMap:
            sat_map0 = SatMap.convert('RGB')
        cx, cy = SatMap.width/2, SatMap.height/2
        sat0 = np.array(sat_map0)
        sat0 = cv2.circle(sat0, center=(int(cx), int(cy)), radius=3, thickness=1,
                          color=(0, 0, 255))

        # 添加误差
        shift_range = self.shift_range_pixel
        gt_shift_x = x_rand * shift_range  # 像方中心按照像方坐标系偏移
        gt_shift_y = y_rand * shift_range  #
        width_raw = height_raw = self.SatMap_process_sidelength
        sat_map0 = \
            sat_map0.transform(
                sat_map0.size, PIL.Image.AFFINE,
                (1, 0, -gt_shift_x,
                 0, 1, -gt_shift_y),
                resample=PIL.Image.BILINEAR)
        sat1 = np.array(sat_map0)
        sat1 = cv2.circle(sat1, center=(int(cx+gt_shift_x), int(cy+gt_shift_y)), radius=3, thickness=1,
                          color=(0, 0, 255))
        random_ori = o_rand * self.rotation_range

        # 读取全景切片数据，并进行方向初始化
        slice_folder = osp.join(self.pro_root, city, self.slice_flag, pano_flag)
        slice_num = self.slice_num
        grd_left_imgs = torch.tensor([])
        sat_imgs = torch.tensor([])
        for i in range(slice_num):
            slice_ind = int(round(self.direct_num/slice_num*i))
            grd_path = osp.join(slice_folder, f"{str(slice_ind)}.jpg")
            with PIL.Image.open(grd_path, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)
                grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
            heading = 90 + i*(360/slice_num)
            if heading > 360:
                heading -= 360
            sat_map = sat_map0.rotate(heading)
            sat2 = np.array(sat_map)
            sat_map = TF.center_crop(sat_map, self.SatMap_process_sidelength)
            # transform
            if self.satmap_transform is not None:
                sat_map = self.satmap_transform(sat_map)
            sat_imgs = torch.cat([sat_imgs, sat_map.unsqueeze(0)], dim=0)

        _, _, height, width = sat_imgs.shape
        gt_y = height / 2 + np.round(gt_shift_y / height_raw * height)
        gt_x = width / 2 + np.round(gt_shift_x / width_raw * width)

        gt_xy = np.array([gt_x, gt_y])
        orientation_angle = 90 + random_ori
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360
        orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi / 180))
        orientation_map[1, :, :] = np.sin(orientation_angle * np.pi / 180)

        return sat_imgs, grd_left_imgs, city, gt_xy, orientation_map, orientation_angle, pano_flag

# 用于估计图像3DoF
# 加载模型对切片图像的3DoF射线估计结果
class PoseEstimation(Dataset):
    def __init__(self, root, pro_dir, pose_dir,
                 rotation_range=10, sat_resize=512, cross_area=True, city_list=None):
        self.root = root
        self.pro_root = pro_dir
        self.pose_dir = pose_dir

        self.rotation_range = rotation_range  # in terms of degree
        self.shift_range_pixel = 160 # in terms of pixels
        self.sat_resize = sat_resize

        self.sat_dir = 'sat_img'
        self.pose_file = 'Pose3DoF'

        self.SatMap_process_sidelength = 640 # 添加误差后中心裁切的图像的大小
        self.meter_per_pixel_dict = {}

        self.image_list = []
        self.citys = []
        skymap_loder = SkyMap_Loader(pro_dir, False, cross_area, False)
        if city_list is not None:
            self.city_list = city_list
        else:
            self.city_list = skymap_loder.get_citys()
        print('load citys:', self.city_list)

        self.pose_mats = {}
        for city in self.city_list:
            grd_datas = skymap_loder.run(city)
            self.image_list.extend([data[:-1] for data in grd_datas])
            self.citys.extend([city]*len(grd_datas))
            self.meter_per_pixel_dict[city] = compute_meters_per_pixel(
                osp.join(self.root, city, self.sat_dir),
                self.SatMap_process_sidelength,
                sat_resize
            )

            poses = {}
            data_file = osp.join(self.pose_dir, city, self.pose_file)
            files = [f for f in os.listdir(data_file) if f.endswith('.npy')]
            for file in files:
                data = np.load(osp.join(data_file, file))
                poses[file[:-4]] = data
            self.pose_mats[city] = poses


    def __len__(self):
        return len(self.image_list)

    def get_file_list(self):
        return self.image_list

    def __getitem__(self, idx):
        # 根据编号读取数据并构建模型的输入数据：grd, sat
        grd_data = self.image_list[idx]
        grd_name, o_rand, x_rand, y_rand = grd_data.split()
        o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
        info = decom_pano_name(grd_name)
        pano_flag = info['flag']
        city = self.citys[idx]

        # 添加误差
        shift_range = self.shift_range_pixel
        gt_shift_x = x_rand * shift_range  # 像方中心按照像方坐标系偏移
        gt_shift_y = y_rand * shift_range  #
        width_raw = height_raw = self.SatMap_process_sidelength

        # 生成真值
        random_ori = o_rand * self.rotation_range
        height = width = self.sat_resize
        gt_y = height / 2 + np.round(gt_shift_y / height_raw * height)
        gt_x = width / 2 + np.round(gt_shift_x / width_raw * width)
        gt_xy = np.array([gt_x, gt_y])
        orientation_angle = 90 + random_ori
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360

        # 加载模型的推理结果
        ray_mat = self.pose_mats[city][pano_flag]

        return city, gt_xy, orientation_angle, pano_flag, ray_mat






# ReliableLoc类，使用模型输出的结果进行可信定位与结果保存
class ReliableLoc():
    def __init__(self, root, pro_dir, pose_dir,
                 rotation_range=10, sat_resize=512, cross_area=True, city_list=None):
        self.root = root
        self.pro_root = pro_dir
        self.pose_dir = pose_dir

        self.rotation_range = rotation_range  # in terms of degree
        self.shift_range_pixel = 160 # in terms of pixels
        self.sat_resize = sat_resize

        self.sat_dir = 'sat_img'
        self.pose_file = 'Pose3DoF'
        self.loc_file = 'LocRes'

        self.SatMap_process_sidelength = 640 # 添加误差后中心裁切的图像的大小
        self.meter_per_pixel_dict = {}

        self.image_list = []
        self.citys = []
        skymap_loder = SkyMap_Loader(pro_dir, False, cross_area, False)
        if city_list is not None:
            self.city_list = city_list
        else:
            self.city_list = skymap_loder.get_citys()
        print('load citys:', self.city_list)

        self.H0_model = NullHypothesis(pro_dir=pro_dir)

        self.pose_mats = {}
        self.slice_num = 12 # 实际使用的切片图像个数
        self.direct_num = 12 # 总的切片图像个数
        used_ind = np.linspace(0, self.direct_num, self.slice_num, endpoint=False).astype(int)
        for city in self.city_list:
            grd_datas = skymap_loder.run(city)
            self.image_list.extend([data[:-1] for data in grd_datas])
            self.citys.extend([city]*len(grd_datas))
            self.meter_per_pixel_dict[city] = compute_meters_per_pixel(
                osp.join(self.root, city, self.sat_dir),
                self.SatMap_process_sidelength,
                sat_resize
            )

            poses = {}
            data_file = osp.join(self.pose_dir, city, self.pose_file)
            files = [f for f in os.listdir(data_file) if f.endswith('.npy')]
            for file in files:
                data = np.load(osp.join(data_file, file))
                poses[file[:-4]] = data[used_ind]
            self.pose_mats[city] = poses

    # 多线程并行进行可信定位
    def processing(self):
        H0_model = self.H0_model
        for city in self.city_list:
            print('processing: ', city)
            data_file = osp.join(self.pose_dir, city, self.pose_file)
            files = [f for f in os.listdir(data_file) if f.endswith('.npy')]
            save_dir = osp.join(self.pose_dir, city, self.loc_file)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            sW = self.sat_resize

            num_processes = 20
            def initializer(data_file, save_dir, sW, H0_model):
                global _global_data_file, _global_save_dir, _global_sW, _global_H0_model, _global_city
                _global_data_file = data_file
                _global_save_dir = save_dir
                _global_sW = sW
                _global_H0_model = H0_model

            pool = Pool(processes=num_processes, initializer=initializer,
                        initargs=(data_file, save_dir, sW, H0_model))

            # 并行处理任务
            pool.map(loc_camera, files)
            # 关闭进程池
            pool.close()
            pool.join()

    def save_results(self):
        loc_err_ = []
        loc_err2_ = []

        city_result = {}
        for city, grd_data in zip(self.citys, self.image_list):

            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']

            # 加载真值
            shift_range = self.shift_range_pixel # 添加误差
            gt_shift_x = x_rand * shift_range  # 像方中心按照像方坐标系偏移
            gt_shift_y = y_rand * shift_range  #
            width_raw = height_raw = self.SatMap_process_sidelength

            random_ori = o_rand * self.rotation_range
            height = width = self.sat_resize
            gt_y = height / 2 + np.round(gt_shift_y / height_raw * height)
            gt_x = width / 2 + np.round(gt_shift_x / width_raw * width)
            gt_angle = 90 + random_ori
            if gt_angle < 0:
                gt_angle = gt_angle + 360
            elif gt_angle > 360:
                gt_angle = gt_angle - 360

            # 加载json中位姿估计结果
            json_dir = osp.join(self.pose_dir, city, self.loc_file)
            json_path = osp.join(json_dir, pano_flag+'.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                pose_dict = json.load(f)
            x, y = pose_dict['LocRes']
            nfa = pose_dict['NFA']
            in_ind = pose_dict['Inlier']
            in_num = np.sum(np.array(in_ind))

            # 加载模型推理结果
            ray_mat = self.pose_mats[city][pano_flag]
            slice_oris = ray_mat[:, 2] # 角度为与正北方向的顺时针夹角值
            meter_per_pixel = self.meter_per_pixel_dict[city]
            loc_err = np.sqrt((x-gt_x)**2 + (y-gt_y)**2) * meter_per_pixel # 在512 * 512的scale上进行误差计算
            ori_err = float(np.mean(slice_oris)-gt_angle)
            if np.any(in_ind):
                ori_err2 = float(np.mean(slice_oris[in_ind])-gt_angle)
            else:
                ori_err2 = ori_err
            x2, y2 = pose_dict['ArgRes']
            loc_err2 = np.sqrt((x-gt_x)**2 + (y-gt_y)**2) * meter_per_pixel # 在512 * 512的scale上进行误差计算

            x *= (640/512)
            y *= (640/512)
            # in_num : 内点切片的个数
            res_data = f"{pano_flag} {x:.3f} {y:.3f} {loc_err:.5f} {nfa:.5f} {ori_err:.5f} {loc_err2:.5f} {ori_err2:.5f} {in_num}"
            if not city in city_result:
                city_result[city] = [res_data]
            else:
                city_result[city].append(res_data)

            if nfa < 0:
                loc_err2_.append(loc_err2)
            loc_err_.append(loc_err2)

        loc_err2 = np.array(loc_err2_)
        loc_err = np.array(loc_err_)
        print('ransac loc mean, median: ', np.mean(loc_err), np.median(loc_err))
        print('nfa loc mean, median: ', np.mean(loc_err2), np.median(loc_err2))
        loc_err2 = loc_err
        print('percentage of samples with localization error under 1m, 3m, 5m, 8m, 10m: ',
              np.sum(loc_err2<1)/len(loc_err2),
              np.sum(loc_err2<3)/len(loc_err2),
              np.sum(loc_err2<5)/len(loc_err2),
              np.sum(loc_err2<8)/len(loc_err2),
              np.sum(loc_err2<10)/len(loc_err2))

        # # 保存测试的结果
        for city, res_list in city_result.items():
            save_txt = osp.join(self.pose_dir, city, 'result.txt')  # flag, lxy, err, NFA,
            with open(save_txt, "w+") as file:
                for data in res_list:
                    file.write(data + "\n")

    # 统计模型定位
    def report_loc_error(self):
        slice_err = np.array([])
        loc_var = np.array([])

        city_result = {}
        for city, grd_data in zip(self.citys, self.image_list):

            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']

            # 加载真值
            shift_range = self.shift_range_pixel # 添加误差
            gt_shift_x = x_rand * shift_range  # 像方中心按照像方坐标系偏移
            gt_shift_y = y_rand * shift_range  #
            width_raw = height_raw = self.SatMap_process_sidelength
            height = width = self.sat_resize

            random_ori = o_rand * self.rotation_range
            coor_path = osp.join(self.pro_root, city, 'slice', pano_flag, 'coordinate.npy')
            coor_mat = np.load(coor_path)
            coor_mat = coor_mat + np.array([[gt_shift_x, gt_shift_y]]) - 320
            coor_mat *= (height / height_raw)

            prd_path = osp.join(self.pose_dir, city, self.pose_file, pano_flag+'.npy')
            prd_locs = np.load(prd_path)
            slice_locs = []
            slice_num = prd_locs.shape[0]
            sW = self.sat_resize
            for batch_idx in range(slice_num):
                heading = batch_idx * 360 / slice_num
                heading += 90  # 转变为与正东方向顺时针夹角值
                if heading > 360:
                    heading -= 360
                x, y, _ = prd_locs[batch_idx]
                x, y = rotate_point(x, y, sW / 2, sW / 2, -heading)
                slice_locs.append([x, y])
            slice_locs = np.array(slice_locs)
            dist = np.linalg.norm((slice_locs-coor_mat), axis=1)
            slice_err = np.hstack((slice_err, dist))
            loc_var = np.hstack((loc_var, np.var(dist)))

        statistic_data_pixel(slice_err)
        print('the var value:')
        statistic_data_pixel(loc_var)


    # 保存每个切片的theta_prd theta_gt is_inlier
    def save_ransac_res(self):

        city_result = {}
        res_mats = []
        for city, grd_data in zip(self.citys, self.image_list):

            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']

            # 加载真值
            shift_range = self.shift_range_pixel # 添加误差
            gt_shift_x = x_rand * shift_range  # 像方中心按照像方坐标系偏移
            gt_shift_y = y_rand * shift_range  #
            width_raw = height_raw = self.SatMap_process_sidelength

            random_ori = o_rand * self.rotation_range
            height = width = self.sat_resize
            gt_y = height / 2 + np.round(gt_shift_y / height_raw * height)
            gt_x = width / 2 + np.round(gt_shift_x / width_raw * width)
            gt_angle = 90 + random_ori
            if gt_angle < 0:
                gt_angle = gt_angle + 360
            elif gt_angle > 360:
                gt_angle = gt_angle - 360

            # 加载json中位姿估计结果
            json_dir = osp.join(self.pose_dir, city, self.loc_file)
            json_path = osp.join(json_dir, pano_flag+'.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                pose_dict = json.load(f)
            prd_pt = np.array(pose_dict['LocRes'])

            nfa = pose_dict['NFA']
            in_ind = pose_dict['Inlier']

            # 加载模型推理结果
            sW = 512
            ray_mat = self.pose_mats[city][pano_flag]
            slice_oris = ray_mat[:, 2] # 角度为与正北方向的顺时针夹角值
            prd_locs = ray_mat[:, :2]
            slice_locs = []
            slice_num = prd_locs.shape[0]
            for batch_idx in range(slice_num):
                heading = batch_idx*360/slice_num
                heading += 90 # 转变为与正东方向顺时针夹角值
                if heading > 360:
                    heading -= 360
                x, y = prd_locs[batch_idx]
                x, y = rotate_point(x, y, sW/2, sW/2, -heading)
                slice_locs.append([x, y])
            slice_locs = np.array(slice_locs)

            # 计算拟合的误差
            gt_pt = np.array([gt_x, gt_y])
            oris = gt_angle * np.ones(slice_oris.shape)
            gt_err = loc_util.rotation_error(gt_pt, slice_locs, oris)
            prd_err = loc_util.rotation_error(prd_pt, slice_locs, oris)
            is_inlier = np.zeros(slice_oris.shape)
            is_inlier[in_ind] = 1
            res_mat = np.vstack((gt_err, prd_err, is_inlier)).T

            if nfa < 0:
                res_mats.append(res_mat)
        res_mats = np.vstack(res_mats)
        txt_path = osp.join(self.pose_dir, 'theta_errs.txt')
        np.savetxt(txt_path, res_mats)

    def temp_test(self):
        txt_path = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_Exp/Sydney/result.txt'
        data_mat = np.loadtxt(txt_path)
        theta_errs = data_mat[:, 4]
        pros = []
        for t_err in theta_errs:
            pros.append(self.H0_model.probability2(abs(t_err)))
        pros = np.array(pros)
        pass_ratio = np.sum(pros < 1/6600) / len(pros)
        print('pass_ratio, theory_prob:', pass_ratio, 1/6600)


    # 保存Query_Rand中的theta_err
    # 相机的位置由ransac估计
    def save_results_query_rand(self):

        city_result = {}
        theta_errs = []
        for city, grd_data in zip(self.citys, self.image_list):

            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']

            # 加载真值
            shift_range = self.shift_range_pixel # 添加误差
            gt_shift_x = x_rand * shift_range  # 像方中心按照像方坐标系偏移
            gt_shift_y = y_rand * shift_range  #
            width_raw = height_raw = self.SatMap_process_sidelength

            random_ori = o_rand * self.rotation_range
            height = width = self.sat_resize
            gt_y = height / 2 + np.round(gt_shift_y / height_raw * height)
            gt_x = width / 2 + np.round(gt_shift_x / width_raw * width)
            gt_angle = 90 + random_ori
            if gt_angle < 0:
                gt_angle = gt_angle + 360
            elif gt_angle > 360:
                gt_angle = gt_angle - 360

            # 加载json中位姿估计结果
            json_dir = osp.join(self.pose_dir, city, self.loc_file)
            json_path = osp.join(json_dir, pano_flag+'.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                pose_dict = json.load(f)
            x, y = pose_dict['LocRes']
            nfa = pose_dict['NFA']
            in_ind = pose_dict['Inlier']
            in_num = np.sum(np.array(in_ind))
            theta_errs.append(pose_dict['LocErr'])
        theta_errs = np.array(theta_errs).ravel()
        txt_path = osp.join(self.pose_dir, 'rand_query_err.txt')
        np.savetxt(txt_path, theta_errs)

    # 保存Query_Rand中的theta_err
    # 相机的位置由真值提供
    def save_results_query_rand2(self):
        # 加载各个城市的slice_cross.txt中的数据
        slice_dict = {}
        for city in self.city_list:
            txt_path = osp.join(self.pro_root, city, 'slice_cross.txt')
            with open(txt_path, 'r') as f:
                grd_datas = f.readlines()
            slice_dict = {}
            for grd_data in grd_datas:
                temp = grd_data.split()
                slice_dict[temp[0]] = grd_data


        city_result = {}
        theta_errs = []
        for city, grd_data in zip(self.citys, self.image_list):

            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']

            # 加载真值
            shift_range = self.shift_range_pixel # 添加误差
            gt_shift_x = x_rand * shift_range  # 像方中心按照像方坐标系偏移
            gt_shift_y = y_rand * shift_range  #
            width_raw = height_raw = self.SatMap_process_sidelength

            random_ori = o_rand * self.rotation_range
            height = width = self.sat_resize
            cam_y = height / 2 + np.round(gt_shift_y / height_raw * height)
            cam_x = width / 2 + np.round(gt_shift_x / width_raw * width)
            cam_xy = np.array([cam_x, cam_y])
            gt_angle = 90 + random_ori
            if gt_angle < 0:
                gt_angle = gt_angle + 360
            elif gt_angle > 360:
                gt_angle = gt_angle - 360

            # 加载模型定位结果
            pose_path = osp.join(self.pose_dir, city, self.pose_file, pano_flag+'.npy')
            pose_mat = np.load(pose_path)
            for i in range(len(pose_mat)):
                slx, sly, _ = pose_mat[i]
                heading = i * 360 / len(pose_mat) + 90
                if heading > 360:
                    heading -= 360
                slx, sly = rotate_point(slx, sly, 256, 256, -heading)

                i_name = osp.join(pano_flag, str(i) + '.jpg')
                slice_line = slice_dict[i_name]
                x = (float(slice_line.split()[-2])-640)/640*512 + 256
                y = (float(slice_line.split()[-2])-640)/640*512 + 256
                gt_xy = np.array([x, y])

                vec_gt = gt_xy - cam_xy
                vec_0 = np.array([0, -1])
                theta_gt = loc_util.vector_angle(vec_0, vec_gt)
                vec_prd = np.array([slx, sly]) - cam_xy
                theta_prd = loc_util.vector_angle(vec_0, vec_prd)
                theta_err = np.degrees(theta_gt - theta_prd)
                theta_errs.append(theta_err)
        theta_errs = np.array(theta_errs).ravel()
        txt_path = osp.join(self.pose_dir, 'rand_query_err_weakly.txt')
        np.savetxt(txt_path, theta_errs)

# ReliableLoc类，使用模型输出的结果进行可信定位与结果保存
class ReliableLoc_Weakly(Dataset):
    def __init__(self, root, pro_dir, pose_dir=None,
                 transform=None, rotation_range=10, sat_resize=512, cross_area=True, city_list=None):
        self.root = root
        self.pro_root = pro_dir
        self.pose_dir = pose_dir

        self.rotation_range = rotation_range  # in terms of degree
        self.shift_range_pixel = 160 # in terms of pixels
        self.sat_resize = sat_resize

        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'
        self.slice_flag = 'slice'
        self.sat_dir = 'sat_img'
        self.pose_file = 'Pose3DoF'
        self.loc_file = 'LocRes'

        self.SatMap_process_sidelength = 640 # 添加误差后中心裁切的图像的大小
        self.meter_per_pixel_dict = {}

        self.image_list = []
        self.image_list_test = []
        self.citys = []
        skymap_loder = SkyMap_Loader(pro_dir, False, cross_area, False)
        self.city_list = ['Chicago', 'Sydney', 'Johannesburg']
        print('load citys:', self.city_list)

        self.H0_model = NullHypothesis(pro_dir=pro_dir)

        self.pose_mats = {}
        self.slice_num = 12 # 实际使用的切片图像个数
        self.direct_num = 12 # 总的切片图像个数
        used_ind = np.linspace(0, self.direct_num, self.slice_num, endpoint=False).astype(int)
        train_ratio = 0.8
        random.seed(2025)
        for city in self.city_list:
            grd_datas = skymap_loder.run(city)

            random.shuffle(grd_datas)
            length = len(grd_datas)
            grd_datas_train = grd_datas[:int(length*train_ratio)] # 取前80%的数据用于生成伪标签
            grd_datas_test = grd_datas[int(length*train_ratio):] # 取所有数据用于生成伪标签
            self.image_list.extend([data[:-1] for data in grd_datas_train])
            self.image_list_test.extend([data[:-1] for data in grd_datas])
            self.citys.extend([city]*len(grd_datas_train))
            self.meter_per_pixel_dict[city] = compute_meters_per_pixel(
                osp.join(self.root, city, self.sat_dir),
                self.SatMap_process_sidelength,
                sat_resize
            )

            if not self.pose_dir is None: # 已生成模型对slice定位的结果
                poses = {}
                data_file = osp.join(self.pose_dir, city, self.pose_file)
                files = [f for f in os.listdir(data_file) if f.endswith('.npy')]
                for file in files:
                    data = np.load(osp.join(data_file, file))
                    poses[file[:-4]] = data[used_ind]
                self.pose_mats[city] = poses

        self.city_dict = {'CHG': 'Chicago',
                          'JNB': 'Johannesburg',
                          'LD': 'London',
                          'RIO': 'Rio',
                          'SD': 'Sydney',
                          'TK': 'Tokyo'}

    def __len__(self):
        return len(self.image_list_test)

    # 加载cross中的测试集
    def __getitem__(self, idx):
        # 根据编号读取数据并构建模型的输入数据：grd, sat
        grd_data = self.image_list_test[idx]
        grd_name, o_rand, x_rand, y_rand = grd_data.split()
        o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
        info = decom_pano_name(grd_name)
        pano_flag = info['flag']
        city = self.city_dict[info['city']]

        # 读取卫星数据并处理
        sat_path = osp.join(self.root, city, self.sat_dir, pano_flag, 'satellite.jpg')
        with PIL.Image.open(sat_path, 'r') as SatMap:
            sat_map0 = SatMap.convert('RGB')
        cx, cy = SatMap.width/2, SatMap.height/2
        sat0 = np.array(sat_map0)
        sat0 = cv2.circle(sat0, center=(int(cx), int(cy)), radius=3, thickness=1,
                          color=(0, 0, 255))

        # 添加误差
        shift_range = self.shift_range_pixel
        gt_shift_x = x_rand * shift_range  # 像方中心按照像方坐标系偏移
        gt_shift_y = y_rand * shift_range  #
        width_raw = height_raw = self.SatMap_process_sidelength
        sat_map0 = \
            sat_map0.transform(
                sat_map0.size, PIL.Image.AFFINE,
                (1, 0, -gt_shift_x,
                 0, 1, -gt_shift_y),
                resample=PIL.Image.BILINEAR)
        sat1 = np.array(sat_map0)
        sat1 = cv2.circle(sat1, center=(int(cx+gt_shift_x), int(cy+gt_shift_y)), radius=3, thickness=1,
                          color=(0, 0, 255))
        random_ori = o_rand * self.rotation_range

        # 读取全景切片数据，并进行方向初始化
        slice_folder = osp.join(self.pro_root, city, self.slice_flag, pano_flag)
        slice_num = self.slice_num
        grd_left_imgs = torch.tensor([])
        sat_imgs = torch.tensor([])
        for i in range(slice_num):
            slice_ind = int(round(self.direct_num/slice_num*i))
            grd_path = osp.join(slice_folder, f"{str(slice_ind)}.jpg")
            with PIL.Image.open(grd_path, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)
                grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
            heading = 90 + i*(360/slice_num)
            if heading > 360:
                heading -= 360
            sat_map = sat_map0.rotate(heading)
            sat2 = np.array(sat_map)
            sat_map = TF.center_crop(sat_map, self.SatMap_process_sidelength)
            # transform
            if self.satmap_transform is not None:
                sat_map = self.satmap_transform(sat_map)
            sat_imgs = torch.cat([sat_imgs, sat_map.unsqueeze(0)], dim=0)

        _, _, height, width = sat_imgs.shape
        gt_y = height / 2 + np.round(gt_shift_y / height_raw * height)
        gt_x = width / 2 + np.round(gt_shift_x / width_raw * width)

        gt_xy = np.array([gt_x, gt_y])
        orientation_angle = 90 + random_ori
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360
        orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi / 180))
        orientation_map[1, :, :] = np.sin(orientation_angle * np.pi / 180)

        return sat_imgs, grd_left_imgs, city, gt_xy, orientation_map, orientation_angle, pano_flag

    # 多线程并行进行可信定位
    def processing(self):
        H0_model = self.H0_model
        for city in self.city_list:
            print('processing: ', city)
            data_file = osp.join(self.pose_dir, city, self.pose_file)
            files = [f for f in os.listdir(data_file) if f.endswith('.npy')]
            save_dir = osp.join(self.pose_dir, city, self.loc_file)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            sW = self.sat_resize

            num_processes = 20
            def initializer(data_file, save_dir, sW, H0_model):
                global _global_data_file, _global_save_dir, _global_sW, _global_H0_model, _global_city
                _global_data_file = data_file
                _global_save_dir = save_dir
                _global_sW = sW
                _global_H0_model = H0_model

            pool = Pool(processes=num_processes, initializer=initializer,
                        initargs=(data_file, save_dir, sW, H0_model))

            # 并行处理任务
            pool.map(loc_camera, files)
            # 关闭进程池
            pool.close()
            pool.join()

    def save_results(self):

        city_result = {}
        for grd_data in self.image_list_test:

            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']
            city = self.city_dict[info['city']]

            # 加载真值
            shift_range = self.shift_range_pixel # 添加误差
            gt_shift_x = x_rand * shift_range  # 像方中心按照像方坐标系偏移
            gt_shift_y = y_rand * shift_range  #
            width_raw = height_raw = self.SatMap_process_sidelength

            random_ori = o_rand * self.rotation_range
            height = width = self.sat_resize
            gt_y = height / 2 + np.round(gt_shift_y / height_raw * height)
            gt_x = width / 2 + np.round(gt_shift_x / width_raw * width)
            gt_angle = 90 + random_ori
            if gt_angle < 0:
                gt_angle = gt_angle + 360
            elif gt_angle > 360:
                gt_angle = gt_angle - 360

            # 加载json中位姿估计结果
            json_dir = osp.join(self.pose_dir, city, self.loc_file)
            json_path = osp.join(json_dir, pano_flag+'.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                pose_dict = json.load(f)
            x, y = pose_dict['LocRes']
            nfa = pose_dict['NFA']
            in_ind = pose_dict['Inlier']
            in_num = np.sum(np.array(in_ind))

            # 加载模型推理结果
            ray_mat = self.pose_mats[city][pano_flag]
            slice_oris = ray_mat[:, 2] # 角度为与正北方向的顺时针夹角值
            meter_per_pixel = self.meter_per_pixel_dict[city]
            loc_err = np.sqrt((x-gt_x)**2 + (y-gt_y)**2) * meter_per_pixel # 在512 * 512的scale上进行误差计算
            ori_err = float(np.mean(slice_oris)-gt_angle)
            if np.any(in_ind):
                ori_err2 = float(np.mean(slice_oris[in_ind])-gt_angle)
            else:
                ori_err2 = ori_err
            x2, y2 = pose_dict['ArgRes']
            loc_err2 = np.sqrt((x2-gt_x)**2 + (y2-gt_y)**2) * meter_per_pixel # 在512 * 512的scale上进行误差计算

            x *= (640/512)
            y *= (640/512)
            # in_num : 内点切片的个数
            res_data = f"{pano_flag} {x:.3f} {y:.3f} {loc_err:.5f} {nfa:.5f} {ori_err:.5f} {loc_err2:.5f} {ori_err2:.5f} {in_num}"
            if not city in city_result:
                city_result[city] = [res_data]
            else:
                city_result[city].append(res_data)

        # # 保存测试的结果
        for city, res_list in city_result.items():
            save_txt = osp.join(self.pose_dir, city, 'result.txt')  # flag, lxy, err, NFA,
            with open(save_txt, "w+") as file:
                for data in res_list:
                    file.write(data + "\n")

    # 根据定位结果生成伪标签——Slice-Loc方法
    # 保存到每个城市的slice_cross_pseudo.txt中
    def generate_pseudo_label(self):
        # 只使用可信定位的内点 + 筛选过的外点
        # 加载模型定位结果、可信判定结果
        train_dict = {}
        for grd_data in self.image_list:
            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']
            city = self.city_dict[info['city']]

            # 加载模型推理结果
            loc_data = self.pose_mats[city][pano_flag]

            # 加载json中位姿估计结果
            json_dir = osp.join(self.pose_dir, city, self.loc_file)
            json_path = osp.join(json_dir, pano_flag+'.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                pose_dict = json.load(f)
            x, y = pose_dict['LocRes']
            nfa = pose_dict['NFA']
            in_ind = pose_dict['Inlier']
            theta_err = pose_dict['LocErr']
            # 保存可信定位的结果
            if nfa > 0:
                continue

            locxy = []
            for i in range(len(in_ind)):
                slx, sly = loc_data[i, 0], loc_data[i, 1]
                heading = i * 360 / len(in_ind) + 90
                if heading > 360:
                    heading -= 360
                slx, sly = rotate_point(slx, sly, 256, 256, -heading)
                slx *= 640/512; slx = slx-320+640
                sly *= 640/512; sly = sly-320+640
                if in_ind[i] or theta_err[i] < 30:
                    locxy.append([str(i), slx, sly]) # i, x, y

            if city not in train_dict:
                train_dict[city] = {pano_flag:locxy}
            else:
                train_dict[city][pano_flag] = locxy

        for city, datas in train_dict.items():
            # 加载slice_corss.txt文件并生成新的slice_cross_pseudo.txt
            txt_path = osp.join(self.pro_root, city, 'slice_cross.txt')
            with open(txt_path, 'r') as f:
                grd_datas = f.readlines()
            slice_dict = {}
            for grd_data in grd_datas:
                temp = grd_data.split()
                slice_dict[temp[0]] = grd_data

            slice_datas = []
            locxys = train_dict[city]
            for p_flag, lxy in locxys.items():
                for i, x, y in lxy:
                    i_name = osp.join(p_flag, i+'.jpg')
                    slice_line = slice_dict[i_name]
                    x -= 160 * float(slice_line.split()[2])
                    y -= 160 * float(slice_line.split()[3])
                    line = f"{slice_line[:-1]} {x:.3f} {y:.3f}"
                    slice_datas.append(line)

            pseudo_dir = osp.join(self.pro_root, 'Pseudo', city)
            if not osp.exists(pseudo_dir):
                os.makedirs(pseudo_dir)
            save_txt = txt_path = osp.join(pseudo_dir, 'slice_cross_pseudo.txt')
            random.shuffle(slice_datas)
            with open(save_txt, "w+") as file:
                for line in slice_datas:
                    file.write(line + "\n")

    # 只使用ransac， 不使用nfa筛选可信的结果
    # 保存到每个城市的slice_cross_ransac.txt中
    def generate_pseudo_label2(self):
        # 只使用ransac
        # 加载模型定位结果、可信判定结果
        train_dict = {}
        for grd_data in self.image_list:
            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']
            city = self.city_dict[info['city']]

            # 加载模型推理结果
            loc_data = self.pose_mats[city][pano_flag]

            # 加载json中位姿估计结果
            json_dir = osp.join(self.pose_dir, city, self.loc_file)
            json_path = osp.join(json_dir, pano_flag+'.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                pose_dict = json.load(f)
            x, y = pose_dict['LocRes']
            nfa = pose_dict['NFA']
            in_ind = pose_dict['Inlier']
            theta_err = pose_dict['LocErr']

            # 只使用粗差剔除结果
            locxy = []
            for i in range(len(in_ind)):
                slx, sly = loc_data[i, 0], loc_data[i, 1]
                heading = i * 360 / len(in_ind) + 90
                if heading > 360:
                    heading -= 360
                slx, sly = rotate_point(slx, sly, 256, 256, -heading)
                slx *= 640/512; slx = slx-320+640
                sly *= 640/512; sly = sly-320+640
                if in_ind[i] or theta_err[i] < 30:
                    locxy.append([str(i), slx, sly]) # i, x, y

            if city not in train_dict:
                train_dict[city] = {pano_flag:locxy}
            else:
                train_dict[city][pano_flag] = locxy

        for city, datas in train_dict.items():
            # 加载slice_corss.txt文件并生成新的slice_cross_pseudo.txt
            txt_path = osp.join(self.pro_root, city, 'slice_cross.txt')
            with open(txt_path, 'r') as f:
                grd_datas = f.readlines()
            slice_dict = {}
            for grd_data in grd_datas:
                temp = grd_data.split()
                slice_dict[temp[0]] = grd_data

            slice_datas = []
            locxys = train_dict[city]
            for p_flag, lxy in locxys.items():
                for i, x, y in lxy:
                    i_name = osp.join(p_flag, i+'.jpg')
                    slice_line = slice_dict[i_name]
                    x -= 160 * float(slice_line.split()[2])
                    y -= 160 * float(slice_line.split()[3])
                    line = f"{slice_line[:-1]} {x:.3f} {y:.3f}"
                    slice_datas.append(line)

            pseudo_dir = osp.join(self.pro_root, 'Pseudo', city)
            if not osp.exists(pseudo_dir):
                os.makedirs(pseudo_dir)
            save_txt = txt_path = osp.join(pseudo_dir, 'slice_cross_pseudo_ransac.txt')
            random.shuffle(slice_datas)
            with open(save_txt, "w+") as file:
                for line in slice_datas:
                    file.write(line + "\n")


    # 生成eccv 2024中用于训练备用学生模型的txt文件
    # 保存到每个城市的slice_cross_all.txt中
    def generate_pseudo_label3(self):
        # 只使用ransac
        # 加载模型定位结果、可信判定结果
        train_dict = {}
        for grd_data in self.image_list:
            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']
            city = self.city_dict[info['city']]

            # 加载模型推理结果
            loc_data = self.pose_mats[city][pano_flag]

            # 加载json中位姿估计结果
            json_dir = osp.join(self.pose_dir, city, self.loc_file)
            json_path = osp.join(json_dir, pano_flag+'.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                pose_dict = json.load(f)
            x, y = pose_dict['LocRes']
            nfa = pose_dict['NFA']
            in_ind = pose_dict['Inlier']
            theta_err = pose_dict['LocErr']

            # 保存所有结果
            locxy = []
            for i in range(len(in_ind)):
                slx, sly = loc_data[i, 0], loc_data[i, 1]
                heading = i * 360 / len(in_ind) + 90
                if heading > 360:
                    heading -= 360
                slx, sly = rotate_point(slx, sly, 256, 256, -heading)
                slx *= 640/512; slx = slx-320+640
                sly *= 640/512; sly = sly-320+640
                locxy.append([str(i), slx, sly]) # i, x, y

            if city not in train_dict:
                train_dict[city] = {pano_flag:locxy}
            else:
                train_dict[city][pano_flag] = locxy

        for city, datas in train_dict.items():
            # 加载slice_corss.txt文件并生成新的slice_cross_pseudo.txt
            txt_path = osp.join(self.pro_root, city, 'slice_cross.txt')
            with open(txt_path, 'r') as f:
                grd_datas = f.readlines()
            slice_dict = {}
            for grd_data in grd_datas:
                temp = grd_data.split()
                slice_dict[temp[0]] = grd_data

            slice_datas = []
            locxys = train_dict[city]
            for p_flag, lxy in locxys.items():
                for i, x, y in lxy:
                    i_name = osp.join(p_flag, i+'.jpg')
                    slice_line = slice_dict[i_name]
                    x -= 160 * float(slice_line.split()[2])
                    y -= 160 * float(slice_line.split()[3])
                    line = f"{slice_line[:-1]} {x:.3f} {y:.3f}"
                    slice_datas.append(line)

            pseudo_dir = osp.join(self.pro_root, 'Pseudo', city)
            if not osp.exists(pseudo_dir):
                os.makedirs(pseudo_dir)
            save_txt = txt_path = osp.join(pseudo_dir, 'slice_cross_all.txt')
            random.shuffle(slice_datas)
            with open(save_txt, "w+") as file:
                for line in slice_datas:
                    file.write(line + "\n")

    # 使用eccv 2024中的
    # 对比学生教师的切片定位结果，取前80%生成伪标签
    def generate_pseudo_label4(self):
        teacher_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Exp_res/cross_0.0'
        student_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Exp_res/aux_student'

        # 加载模型定位结果、可信判定结果
        train_dict = {}
        dist = []
        for grd_data in self.image_list:
            grd_name, o_rand, x_rand, y_rand = grd_data.split()
            o_rand, x_rand, y_rand = float(o_rand), float(x_rand), float(y_rand)
            info = decom_pano_name(grd_name)
            pano_flag = info['flag']
            city = self.city_dict[info['city']]

            # 加载模型推理结果
            t_mat = np.load(osp.join(teacher_dir, city, self.pose_file, pano_flag+'.npy'))
            s_mat = np.load(osp.join(student_dir, city, self.pose_file, pano_flag+'.npy'))
            locxy = []
            for i in range(len(t_mat)):
                slx, sly = t_mat[i, 0], t_mat[i, 1]
                lx_, ly_ = s_mat[i, 0], s_mat[i, 1]
                d = (slx-lx_)**2 + (sly-ly_)**2

                heading = i * 360 / len(t_mat) + 90
                if heading > 360:
                    heading -= 360
                slx, sly = rotate_point(slx, sly, 256, 256, -heading)
                slx *= 640 / 512;
                slx = slx - 320 + 640
                sly *= 640 / 512;
                sly = sly - 320 + 640
                locxy.append([str(i), slx, sly, d])  # i, x, y
                dist.append(d)

            if city not in train_dict:
                train_dict[city] = {pano_flag: locxy}
            else:
                train_dict[city][pano_flag] = locxy

        # 根据教师与学生之间的误差，计算误差阈值
        dist = np.array(dist)
        dist.sort()
        ind = int(np.round(0.8 * len(dist)))
        thr = dist[ind]

        for city, datas in train_dict.items():
            # 加载slice_corss.txt文件并生成新的slice_cross_pseudo.txt
            txt_path = osp.join(self.pro_root, city, 'slice_cross.txt')
            with open(txt_path, 'r') as f:
                grd_datas = f.readlines()
            slice_dict = {}
            for grd_data in grd_datas:
                temp = grd_data.split()
                slice_dict[temp[0]] = grd_data

            slice_datas = []
            locxys = train_dict[city]
            for p_flag, lxy in locxys.items():
                for i, x, y, d in lxy:
                    i_name = osp.join(p_flag, i + '.jpg')
                    slice_line = slice_dict[i_name]
                    x -= 160 * float(slice_line.split()[2])
                    y -= 160 * float(slice_line.split()[3])
                    line = f"{slice_line[:-1]} {x:.3f} {y:.3f}"
                    if d <= thr:
                        slice_datas.append(line)

            pseudo_dir = osp.join(self.pro_root, 'Pseudo', city)
            if not osp.exists(pseudo_dir):
                os.makedirs(pseudo_dir)
            save_txt = txt_path = osp.join(pseudo_dir, 'slice_cross_eccv.txt')
            random.shuffle(slice_datas)
            with open(save_txt, "w+") as file:
                for line in slice_datas:
                    file.write(line + "\n")


# 根据flag.npy文件加载推理结果并进行可信定位
def loc_camera(file):
    data_file = _global_data_file
    save_dir = _global_save_dir
    sW = _global_sW
    H0_model = _global_H0_model

    all_num = 12
    use_num = 12
    ind_ = np.linspace(0, all_num, use_num, endpoint=False).astype(int)
    pose_mat = np.load(osp.join(data_file, file))
    pose_mat = pose_mat[ind_]
    slice_num = pose_mat.shape[0]
    prd_locs = pose_mat[:, :2]
    if pose_mat.shape[1] > 2:
        slice_oris = pose_mat[:, 2]  # 角度为与正北方向的顺时针夹角值
    else:
        slice_oris = 90 * np.ones((pose_mat.shape[0], 1))
    prd_ori = np.mean(slice_oris) - 90
    slice_locs = []
    for batch_idx in range(slice_num):
        heading = batch_idx * 360 / slice_num
        heading += 90  # 转变为与正东方向顺时针夹角值
        if heading > 360:
            heading -= 360
        x, y = prd_locs[batch_idx]
        x, y = rotate_point(x, y, sW / 2, sW / 2, -heading)
        slice_locs.append([x, y])
    slice_locs = np.array(slice_locs)
    loc_res = np.mean(slice_locs, axis=0)

    # sloc_err1 在粗差剔除中进行判定的距离
    temp, inlier_idx, NFA, sloc_err1 = \
        loc_util.estimate_camera_pose(slice_locs,
                                      out_nfa=True,
                                      prd_directs=slice_oris,
                                      h0=H0_model)
    res = {
        'ArgRes': loc_res.astype(float).tolist(),
        'LocRes': temp.astype(float).tolist(),
        'Inlier': inlier_idx.tolist(),
        'NFA': NFA[0].astype(float),
        'LocErr': sloc_err1.astype(float).tolist()
    }
    json_path = osp.join(save_dir, file[:-4] + '.json')
    with open(json_path, 'w+', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    pass