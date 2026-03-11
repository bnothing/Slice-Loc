#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil
import torch
import sys
import os.path as osp
import os
import time
import cv2
import math
import numpy as np
from pyproj import Proj, CRS
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import threading
from multiprocessing import Pool
import PIL.Image
import PIL.ImageDraw
import torchvision.transforms.functional as TF


def predict(train_config, model, dataloader, model_path):
    batch_save_path = rf"{model_path}/img_features_batches"

    if not os.path.exists(batch_save_path):
        os.makedirs(batch_save_path)

    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    # -- draw visualization for ablation study
    draw_vis = False
    if draw_vis:
        pic_path = "./draw_vis"
        iterations = int(len(os.listdir(pic_path)) / 3)
        for i in range(iterations):
            bev_ori = cv2.imread(rf"{pic_path}/{i}_bev.jpg")
            pano_ori = cv2.imread(rf"{pic_path}/{i}_pano.jpg")
            sat_ori = cv2.imread(rf"{pic_path}/{i}_sat.png")

            bev_shape = bev_ori.shape[:-1]
            pano_shape = pano_ori.shape[:-1]
            sat_shape = sat_ori.shape[:-1]

            bev = cv2.resize(bev_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            pano = cv2.resize(pano_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            sat = cv2.resize(sat_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0

            bev = torch.tensor(bev).permute(2, 0, 1)  # 通道顺序变换
            pano = torch.tensor(pano).permute(2, 0, 1)
            sat = torch.tensor(sat).permute(2, 0, 1)

            # 图像标准化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(mean=mean, std=std)
            bev = normalize(bev)[None, :, :, :]
            pano = normalize(pano)[None, :, :, :]
            sat = normalize(sat)[None, :, :, :]

            with torch.no_grad():
                with autocast():
                    bev = bev.to(train_config.device)
                    pano = pano.to(train_config.device)
                    sat = sat.to(train_config.device)

                    img_feature_bev = model(bev)[1]
                    # img_feature_bev = F.normalize(img_feature_bev, dim=1)
                    img_feature_pano = model(pano)[1]
                    # img_feature_pano = F.normalize(img_feature_pano, dim=1)
                    img_feature_sat = model(sat)[1]
                    # img_feature_sat = F.normalize(img_feature_sat, dim=1)

                    heat_map_bev = img_feature_bev[0].permute(1, 2, 0)
                    heat_map_bev = torch.mean(heat_map_bev, dim=2).detach().cpu().numpy()
                    heat_map_bev = (heat_map_bev - heat_map_bev.min()) / (heat_map_bev.max() - heat_map_bev.min())
                    heat_map_bev = cv2.resize(heat_map_bev, [bev_shape[1], bev_shape[0]])

                    heat_map_pano = img_feature_pano[0].permute(1, 2, 0)
                    heat_map_pano = torch.mean(heat_map_pano, dim=2).detach().cpu().numpy()
                    heat_map_pano = (heat_map_pano - heat_map_pano.min()) / (heat_map_pano.max() - heat_map_pano.min())
                    heat_map_pano = cv2.resize(heat_map_pano, [pano_shape[1], pano_shape[0]])

                    heat_map_sat = img_feature_sat[0].permute(1, 2, 0)
                    heat_map_sat = torch.mean(heat_map_sat, dim=2).detach().cpu().numpy()
                    heat_map_sat = (heat_map_sat - heat_map_sat.min()) / (heat_map_sat.max() - heat_map_sat.min())
                    heat_map_sat = cv2.resize(heat_map_sat, [sat_shape[1], sat_shape[0]])

                    #  colorize
                    colored_image_bev = cv2.applyColorMap((heat_map_bev * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_pano = cv2.applyColorMap((heat_map_pano * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_sat = cv2.applyColorMap((heat_map_sat * 255).astype(np.uint8), cv2.COLORMAP_JET)

                    # 设置半透明度（alpha值）
                    alpha = 0.5
                    # 将两个图像进行叠加
                    blended_image_bev = cv2.addWeighted(bev_ori, alpha, colored_image_bev, 1 - alpha, 0)
                    blended_image_pano = cv2.addWeighted(pano_ori, alpha, colored_image_pano, 1 - alpha, 0)
                    blended_image_sat = cv2.addWeighted(sat_ori, alpha, colored_image_sat, 1 - alpha, 0)

                    cv2.imwrite(rf"{pic_path}/{i}_bev_vis.jpg", blended_image_bev)
                    cv2.imwrite(rf"{pic_path}/{i}_pano_vis.jpg", blended_image_pano)
                    cv2.imwrite(rf"{pic_path}/{i}_sat_vis.jpg", blended_image_sat)
        return 0

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    ids_list = []
    locs_list = []
    batch_count = 0

    with torch.no_grad():
        for img, ids, locs in bar:
            ids_list.append(ids)
            locs_list.append(torch.cat((locs[0].unsqueeze(1), locs[1].unsqueeze(1)), dim=1))

            with autocast():
                img = img.to(train_config.device)
                img_feature = model(img)[0]

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_feature = img_feature.to(torch.float32)

            # Save the current batch to disk
            torch.save(img_feature, os.path.join(batch_save_path, f'batch_{batch_count}.pt'))
            batch_count += 1

        # Combine ids and locs
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        locs_list = torch.cat(locs_list, dim=0)

    if train_config.verbose:
        bar.close()

    # Load and concatenate all batches from disk
    img_features_list = []
    for i in range(batch_count):
        batch_features = torch.load(os.path.join(batch_save_path, f'batch_{i}.pt'))
        img_features_list.append(batch_features)

    img_features = torch.cat(img_features_list, dim=0)

    # Clean up temporary files
    for i in range(batch_count):
        os.remove(os.path.join(batch_save_path, f'batch_{i}.pt'))

    return img_features, ids_list, locs_list


# theta 为正逆时针旋转，为负顺时针旋转
def rotate_point(x, y, x_c, y_c, theta):
    # 旋转矩阵
    theta = theta * (np.pi / 180)
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


# 统计判定结果
def statistics_valid_res(prd_valids, gt_valids):
    prd_valids = np.array(prd_valids).squeeze()
    gt_valids = np.array(gt_valids).squeeze()
    TP = np.sum((gt_valids == True) & (prd_valids == True))
    FP = np.sum((gt_valids == False) & (prd_valids == True))
    TN = np.sum((gt_valids == False) & (prd_valids == False))
    FN = np.sum((gt_valids == True) & (prd_valids == False))
    print('TP, FP, TN, FN:', TP, FP, TN, FN)
    # RoTP = TP/(TP+FN)# recall of TN
    # print('Recall of TP:', RoTP)
    # if TP>0 or FP>0:
    #     PoTP = TP/(TP+FP)
    #     print('precision of TP:', PoTP) # precision
    # else:
    #     PoTP = 0
    #     print('precision: - ')
    # if PoTP == 0 and RoTP == 0:
    #     print('the F1: -')
    # else:
    #     print('the F1 of TP: ', 2*PoTP*RoTP/(PoTP+RoTP))

    RoTN = TN / (TN + FP)
    print('Recall of TN:', RoTN)
    if TN > 0 or FN > 0:
        PoTN = TN / (TN + FN)
        print('precision of TN:', PoTN)  # precision
    else:
        PoTN = 0
        print('precision of TN: - ')

    # print('Acc:', (TP+TN)/len(gt_valids))
    # round(np.sum(data < 10) / len(data)*100,2):.2f
    return f"{round(RoTN * 100, 2):.2f}"


# =======================================================================================================
# 将全景影像进行切片
def crop_panorama_image(img, theta=0.0, phi=0.0, res_x=512, res_y=512, fov=60.0, debug=False, output_ind=False):
    img_x = img.shape[0]
    img_y = img.shape[1]

    theta = theta / 180 * math.pi
    phi = phi / 180 * math.pi

    fov_x = fov
    aspect_ratio = res_y * 1.0 / res_x
    half_len_x = math.tan(fov_x / 180 * math.pi / 2)
    half_len_y = aspect_ratio * half_len_x

    pixel_len_x = 2 * half_len_x / res_x
    pixel_len_y = 2 * half_len_y / res_y

    map_x = np.zeros((res_x, res_y), dtype=np.float32)
    map_y = np.zeros((res_x, res_y), dtype=np.float32)

    axis_y = math.cos(theta)
    axis_z = math.sin(theta)
    axis_x = 0

    # theta rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    theta_rot_mat = np.array([[1, 0, 0], \
                              [0, cos_theta, -sin_theta], \
                              [0, sin_theta, cos_theta]], dtype=np.float32)

    # phi rotation matrix
    cos_phi = math.cos(phi)
    sin_phi = -math.sin(phi)
    phi_rot_mat = np.array([[cos_phi + axis_x ** 2 * (1 - cos_phi), \
                             axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
                             axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
                            [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
                             cos_phi + axis_y ** 2 * (1 - cos_phi), \
                             axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
                            [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
                             axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
                             cos_phi + axis_z ** 2 * (1 - cos_phi)]], dtype=np.float32)

    map_x = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
    map_y = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

    map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
    map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
    map_z = np.ones((res_x, res_y)).astype(np.float32) * -1

    ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
                                     np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

    ind = theta_rot_mat.dot(ind)
    ind = phi_rot_mat.dot(ind)

    vec_len = np.sqrt(np.sum(ind ** 2, axis=0))
    ind /= np.tile(vec_len, (3, 1))

    cur_phi = np.arcsin(ind[0, :])
    cur_theta = np.arctan2(ind[1, :], -ind[2, :])

    map_x = (cur_phi + math.pi / 2) / math.pi * img_x
    map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

    map_x = np.reshape(map_x, [res_x, res_y])
    map_y = np.reshape(map_y, [res_x, res_y])

    if debug:
        for x in range(res_x):
            for y in range(res_y):
                print('(%.2f, %.2f)\t' % (map_x[x, y], map_y[x, y]))
            print(" ")

    if output_ind:
        return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP), map_y, map_x
    else:
        return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# 解析全景图像名，返回相关信息
def decom_pano_name(pano_path):
    p_name = osp.basename(pano_path)
    i_name, ext = osp.splitext(p_name)
    datas = i_name.split(',')
    res_dict = {
        'ext': ext,
        'i_name': i_name,
        'city': datas[0],
        'flag': datas[1],
        'lat': float(datas[2]),
        'lon': float(datas[3]),
        'hei': float(datas[4]),
        'year': datas[5]
    }
    return res_dict


def decom_pano_name_vigor(pano_path):
    p_name = osp.basename(pano_path)
    i_name, ext = osp.splitext(p_name)
    datas = i_name.split(',')
    res_dict = {
        'ext': ext,
        'i_name': i_name,
        'flag': datas[0],
        'lat': float(datas[1]),
        'lon': float(datas[2])
    }
    return res_dict


def create_file(path):
    if not osp.exists(path):
        os.makedirs(path)


def get_utm_epsg_code(lon, lat):
    # 创建 WGS84 坐标系的 CRS 对象
    wgs84 = CRS("EPSG:4326")
    # 获取经纬度所在的 UTM 区域
    utm_zone = int((lon + 180) / 6) + 1
    # 构建对应的 UTM CRS 对象
    utm_crs = CRS(f"EPSG:326{utm_zone}" if lat >= 0 else f"EPSG:327{utm_zone}")
    return utm_crs.to_epsg()


def get_utm_epsg(longitude, latitude):
    # 确定UTM区域号（取整数部分加1，因为UTM区域从1开始编号）
    utm_zone_number = int((abs(longitude) + 180.0) // 6) + 1
    # 确定南北半球（北半球为'N'，南半球为'S'）
    hemisphere = 'N' if latitude >= 0 else 'S'
    # 构造EPSG代码
    # 对于北半球，以'326'开头；对于南半球，以'327'开头
    epsg_code = f'epsg:{326 + (100 if hemisphere == "S" else 0)}{utm_zone_number:02d}'
    return epsg_code


# 卫星像方<->交会空间
class GeoTrans():
    def __init__(self,
                 json_path=None,
                 tl_path=None,
                 sat_size=1280):
        self.json_path = json_path
        self.tl_path = tl_path
        # 加载json文件中数据
        if not json_path is None:
            with open(json_path, 'r') as f:
                trans_dict = json.load(f)
            self.grd_hei = trans_dict['ground_height']
            self.x_param = np.array(trans_dict['x_param'])
            self.y_param = np.array(trans_dict['y_param'])

        # 设置投影坐标系
        ref_range = np.loadtxt(tl_path, delimiter=' ')
        tl_lon, tl_lat = ref_range[0]
        br_lon, br_lat = ref_range[1]
        size = sat_size
        self.sol_lon = (br_lon - tl_lon) / size
        self.sol_lat = (br_lat - tl_lat) / size
        self.tl_lon = tl_lon
        self.tl_lat = tl_lat

        epsg_code = get_utm_epsg_code(tl_lon, tl_lat)
        self.ToUTM = Proj(epsg_code)
        self.Degree2MeterRatio()

    # 将航空交会空间三维坐标投影转为图像上坐标
    def Geo2Ispace(self, xs, ys, heis):
        lons, lats = self.ToUTM(xs, ys, inverse=True)
        proj_x = (lons - self.tl_lon) / self.sol_lon - 0.5
        proj_y = (lats - self.tl_lat) / self.sol_lat - 0.5
        if self.json_path is None:
            t_x, t_y = proj_x, proj_y
        else:
            A_x = np.stack((heis - self.grd_hei, np.ones_like(heis)))
            t_x = A_x.T @ self.x_param + proj_x
            A_y = np.stack((heis - self.grd_hei, np.ones_like(heis)))
            t_y = A_y.T @ self.y_param + proj_y
        return t_x, t_y

    # 卫星的像方空间转换到物方交会空间
    def Ispace2Geo(self, xs, ys, heis):
        A_x = np.stack((heis - self.grd_hei, np.ones_like(heis)))
        proj_x = xs - A_x.T @ self.x_param
        A_y = np.stack((heis - self.grd_hei, np.ones_like(heis)))
        proj_y = ys - A_y.T @ self.y_param
        lons = (proj_x + 0.5) * self.sol_lon + self.tl_lon
        lats = (proj_y + 0.5) * self.sol_lat + self.tl_lat
        tx, ty = self.ToUTM(lons, lats, inverse=False)
        return tx, ty

    # 进行单位转化，将度转为米
    def Degree2MeterRatio(self):
        x1, y1 = self.ToUTM(self.tl_lon, self.tl_lat)
        x2, y2 = self.ToUTM(self.tl_lon + 0.00001, self.tl_lat)
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        self.lon_ratio = dist / 0.00001
        x3, y3 = self.ToUTM(self.tl_lon, self.tl_lat + 0.00001)
        dist = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
        self.lat_ratio = dist / 0.00001


# 根据深度与像方坐标生成物方坐标
def find_coor(depth, map_xy, trans, pinfo, depth_scale=0.2):
    default_dist = 80  # 没有数据满足要求时，默认距离
    # 转角度
    ui, vi = map_xy[:, :, 0] / 4, map_xy[:, :, 1] / 4
    ui, vi = np.ravel(ui), np.ravel(vi)
    u = (ui / (depth.shape[1])) * 2 * np.pi - np.pi
    v = (vi / (depth.shape[0])) * np.pi
    ui = ui.astype(int)
    vi = vi.astype(int)

    # 选取深度
    vi[vi >= depth.shape[0]] = depth.shape[0] - 1
    ui[ui >= depth.shape[1]] = depth.shape[1] - 1
    Z = depth[vi, ui, 0]
    valid = Z < 255.  # 筛选出有效的角度
    if np.sum(valid) <= 0:
        return None

    theta = u[valid]  # x
    phi = v[valid]  # y
    Z = Z[valid]
    Z *= depth_scale

    # 坐标生成
    lon0, lat0, hei0 = pinfo['lon'], pinfo['lat'], pinfo['hei']
    x0, y0 = trans.ToUTM(lon0, lat0)

    dir_x = np.sin(phi) * np.sin(theta)
    dir_y = np.sin(phi) * np.cos(theta)
    dir_z = np.cos(phi)

    x_s = lon0 + (Z * dir_x) / trans.lon_ratio
    y_s = lat0 + (Z * dir_y) / trans.lat_ratio
    xs, ys = trans.ToUTM(x_s, y_s)
    heis = hei0 + Z * dir_z

    dist = 20
    x_ind = np.abs(xs - x0) < dist
    y_ind = np.abs(ys - y0) < dist
    # z_ind = heis > 0
    r_ind = x_ind & y_ind
    if np.sum(r_ind) <= 0:
        return None
    res = np.stack((xs, ys, heis), axis=1)
    res = res[r_ind]
    ix, iy = trans.Geo2Ispace(res[:, 0], res[:, 1], res[:, 2])
    return np.stack((ix, iy), axis=1)


def find_coor2(depth, map_xy, trans, pinfo, depth_scale=0.2):
    ind_ = np.arange(0, map_xy.shape[0])
    X, Y = np.meshgrid(ind_, ind_)
    X, Y = np.ravel(X), np.ravel(Y)
    ind_ = np.stack((X, Y), axis=1)

    default_dist = 80  # 没有数据满足要求时，默认距离
    # 转角度
    ui, vi = map_xy[:, :, 0] / 4, map_xy[:, :, 1] / 4
    ui, vi = np.ravel(ui), np.ravel(vi)

    u = (ui / (depth.shape[1])) * 2 * np.pi - np.pi
    v = (vi / (depth.shape[0])) * np.pi
    ui = ui.astype(int)
    vi = vi.astype(int)

    # 选取深度
    vi[vi >= depth.shape[0]] = depth.shape[0] - 1
    ui[ui >= depth.shape[1]] = depth.shape[1] - 1
    Z = depth[vi, ui, 0]
    valid = Z < 255.  # 筛选出有效的角度
    if np.sum(valid) <= 0:
        return None

    theta = u[valid]
    phi = v[valid]
    Z = Z[valid]
    ind_ = ind_[valid]
    Z *= depth_scale

    # 坐标生成
    lon0, lat0, hei0 = pinfo['lon'], pinfo['lat'], pinfo['hei']
    x0, y0 = trans.ToUTM(lon0, lat0)

    dir_x = np.sin(phi) * np.sin(theta)
    dir_y = np.sin(phi) * np.cos(theta)
    dir_z = np.cos(phi)

    x_s = lon0 + (Z * dir_x) / trans.lon_ratio
    y_s = lat0 + (Z * dir_y) / trans.lat_ratio
    xs, ys = trans.ToUTM(x_s, y_s)
    heis = hei0 + Z * dir_z

    dist = 20
    x_ind = np.abs(xs - x0) < dist
    y_ind = np.abs(ys - y0) < dist
    # z_ind = heis > 0
    r_ind = x_ind & y_ind
    if np.sum(r_ind) <= 0:
        return None
    res = np.stack((xs, ys, heis), axis=1)
    res = res[r_ind]
    ix, iy = trans.Geo2Ispace(res[:, 0], res[:, 1], res[:, 2])
    ind_ = ind_[r_ind]
    return np.stack((ix, iy), axis=1), ind_


# 处理：切分全景，根据深度生成训练真值
def main(argv):
    # 遍历全景并切块，生成对应的txt列表
    slice_num = 16  # 默认
    down_angle = 45  # 向下观察的角度
    default_dist = 80  # 默认为80个像素

    city = 'Chicago'
    slice_flag = 'slice'
    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    pano_dir = osp.join(raw_dir, city, 'panorama')
    sat_dir = osp.join(raw_dir, city, 'sat_img')
    output_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing'  # 保存路径
    depth_dir = osp.join(raw_dir, city, 'depth', 'North')

    # 创建相关的文件夹
    output_file = osp.join(output_dir, city)
    create_file(output_file)
    slice_file = osp.join(output_file, slice_flag)
    create_file(slice_file)

    # 进行切片并保存图片、对应矩阵
    files = [f for f in os.listdir(pano_dir) if f.endswith('.jpg')]

    # 输出文件夹，用于保存裁剪后的图像
    os.makedirs(output_dir, exist_ok=True)
    resolution_x = 512
    resolution_y = 512
    fov = 90.0
    phi = 0.0
    for file in tqdm(files):
        file_path = os.path.join(pano_dir, file)
        img_info = decom_pano_name(file_path)
        depth_path = osp.join(depth_dir, img_info['i_name'] + '.depthmap.jpg')

        p_path = osp.join(slice_file, img_info['flag'])
        create_file(p_path)
        coor_path = osp.join(slice_file, img_info['flag'], 'coordinate.npy')

        # 创建坐标转换的类
        sat_path = osp.join(sat_dir, img_info['flag'], 'satellite.jpg')
        sat_size = cv2.imread(sat_path).shape[0]
        tl_path = osp.join(sat_dir, img_info['flag'], 'tl_pos.txt')
        i_trans = GeoTrans(json_path=None, tl_path=tl_path, sat_size=sat_size)
        sat_img = cv2.imread(sat_path)

        img = cv2.imread(file_path)
        depth = cv2.imread(depth_path).astype('float64')
        coor_list = []
        for i in range(slice_num):  # 对每个切片
            theta = i * 360 / slice_num
            out_img, map_w1, map_h1 = crop_panorama_image(img, theta=theta, phi=phi, res_x=resolution_x, \
                                                          res_y=resolution_y, fov=fov, debug=False, output_ind=True)
            map_xy = np.dstack((map_w1, map_h1))
            sat_ips = find_coor(depth, map_xy, i_trans, img_info)
            if sat_ips is None:
                # 评估视觉地标空间距离失败，使用默认的距离
                cx, cy = sat_img.shape[1] / 2, sat_img.shape[0] / 2
                sat_ips = rotate_point(cx, cy + default_dist, cx, cy, -theta)
                sat_ips = np.array(sat_ips)[None]
                print('use default distance:', img_info['flag'])

            mv = np.mean(sat_ips, axis=0)
            # sat_img = cv2.circle(sat_img, center=(int(mv[0]), int(mv[1])), radius=3, thickness=1,
            #                      color=(0, 0, 255))
            coor_list.append(mv)
            output_file_path = os.path.join(p_path, str(i) + '.jpg')

            # 保存提取出的图像
            cv2.imwrite(output_file_path, out_img)
        # temp_path = osp.join(p_path, 'sat_res.jpg') # 保存定位在卫星上的投影
        # cv2.imwrite(temp_path, sat_img)
        coor_mat = np.array(coor_list)
        np.save(coor_path, coor_mat)


# 打乱训练数据的顺序，并保存在txt中
# 生成训练与验证数据
def main2(argv):
    train_ratio = 0.8  # 将前80%的数据作为训练数据
    slice_num = 12
    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing'
    city = 'Sydney'
    slice_flag = 'slice'
    pano_flag = 'panorama'

    train_path = osp.join(pro_dir, city, 'train.txt')
    test_path = osp.join(pro_dir, city, 'test.txt')
    location_path = osp.join(pro_dir, city, 'location_test.txt')
    location_path2 = osp.join(pro_dir, city, 'location_train.txt')  # 保存用于训练的flag

    pro_path = osp.join(pro_dir, city, slice_flag)
    pano_dir = osp.join(raw_dir, city, pano_flag)
    files = [f for f in os.listdir(pano_dir) if f.endswith('.jpg')]

    # 打乱全景的顺序，划分训练的全景与测试的全景
    random.seed(100)
    rand_ind = list(range(len(files)))
    random.shuffle(rand_ind)
    train_num = int(train_ratio * len(rand_ind))
    train_ind = rand_ind[0:train_num]
    test_ind = rand_ind[train_num:]

    # 生成训练的txt文件
    train_img_list = []
    train_coor_list = []
    for ind in train_ind:
        img_name = files[ind]
        info = decom_pano_name(img_name)
        file = info['flag']
        slice_path = osp.join(pro_path, file)
        imgs = [osp.join(file, f"{i}.jpg") for i in range(slice_num)]
        train_img_list.extend(imgs)

        coor_path = osp.join(slice_path, 'coordinate.npy')
        coors = np.load(coor_path).tolist()
        train_coor_list.extend(coors)
    temp = list(range(len(train_img_list)))
    random.shuffle(temp)
    temp = temp[0:int(len(temp) / 2)]
    with open(train_path, "w+") as file:
        for i in temp:
            line = f"{train_img_list[i]} {train_coor_list[i][0]:.3f} {train_coor_list[i][1]:.3f}"
            file.write(line + "\n")

    # 生成测试的txt文件
    test_img_list = []
    test_coor_list = []
    for ind in test_ind:
        img_name = files[ind]
        info = decom_pano_name(img_name)
        file = info['flag']
        slice_path = osp.join(pro_path, file)
        imgs = [osp.join(file, f"{i}.jpg") for i in range(slice_num)]
        test_img_list.extend(imgs)

        coor_path = osp.join(slice_path, 'coordinate.npy')
        coors = np.load(coor_path).tolist()
        test_coor_list.extend(coors)
    temp = list(range(len(test_img_list)))
    random.shuffle(temp)
    with open(test_path, "w+") as file:
        for i in temp:
            line = f"{test_img_list[i]} {test_coor_list[i][0]:.3f} {test_coor_list[i][1]:.3f}"
            file.write(line + "\n")

    # 生成用于定位的txt文件
    train_ind = train_ind[0:int(len(train_ind) / 2)]
    with open(location_path2, "w+") as file:
        for ind in train_ind:
            file.write(files[ind] + "\n")
    with open(location_path, "w+") as file:
        for ind in test_ind:
            file.write(files[ind] + "\n")


# 生成用于全景影像整张定位的文本列表
def main3(argv):
    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing'
    city = 'Chicago'
    slice_flag = 'slice'

    # 用于加载数据的txt文件路径
    location_path = osp.join(pro_dir, city, 'location_test.txt')

    pano_dir = osp.join(raw_dir, city, 'panorama')
    txt_path = osp.join(pro_dir, city, 'location_test.txt')
    files = [f for f in os.listdir(pano_dir) if f.endswith('.jpg')]

    num = 400
    num = min(num, len(files))
    select_list = files[:num]
    with open(txt_path, "w+") as file:
        for name in select_list:
            file.write(name + "\n")


# 对vigor数据进行切片，用于测试精度
def main4(argv):
    # 遍历全景并切块，生成对应的txt列表
    slice_num = 12
    down_angle = 45  # 向下观察的角度
    crop_pano = False

    # 'Chicago', 'NewYork', 'SanFrancisco', 'Seattle'
    city = 'SanFrancisco'
    slice_flag = 'slice'
    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/VIGOR'
    pano_dir = osp.join(raw_dir, city, 'panorama')
    output_dir = '/media/xmt/新加卷/Data/VIGOR_Processing'

    location_path = osp.join(output_dir, city, 'location_test.txt')
    # 创建相关的文件夹
    output_file = osp.join(output_dir, city)
    create_file(output_file)
    slice_file = osp.join(output_file, slice_flag)
    create_file(slice_file)

    # 进行切片并保存图片、对应矩阵
    files_ = [f for f in os.listdir(pano_dir) if f.endswith('.jpg')]
    random.seed(10)
    rand_ind = list(range(len(files_)))
    random.shuffle(rand_ind)
    files = files_

    # 输出文件夹，用于保存裁剪后的图像
    os.makedirs(output_dir, exist_ok=True)

    if crop_pano:
        num_processes = 22

        def initializer(city, raw_dir, output_dir):
            global _global_city, _global_raw_dir, _global_output_dir
            _global_city = city
            _global_raw_dir = raw_dir
            _global_output_dir = output_dir

        pool = Pool(processes=num_processes, initializer=initializer, initargs=(city, raw_dir, output_dir))

        # 并行处理任务
        pool.map(slice_pano_image_vigor, files_)
        # 关闭进程池
        pool.close()
        pool.join()

    # 打乱全景的顺序，划分训练的全景与测试的全景
    random.seed(100)
    rand_ind = list(range(len(files)))
    random.shuffle(rand_ind)

    # 生成用于定位的txt文件
    with open(location_path, "w+") as file:
        for ind in rand_ind:
            i_name = files[ind]
            temp = decom_pano_name_vigor(i_name)
            file.write(temp['flag'] + "\n")


# 生成用于测试NFA有效性的txt列表文件
def main5(argv):
    # 读取用于定位的地面影像
    # 取前50%为出现了地面场景，后50%为负样本
    positive_ratio = 0.5
    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_45'
    city = 'Chicago'
    slice_flag = 'slice'
    pano_flag = 'panorama'

    location_path = osp.join(pro_dir, city, 'same_area_balanced_train.txt')
    nfa_valid_path = osp.join(pro_dir, city, 'half_select_sat2.txt')

    location_imgs = []
    with open(location_path, 'r') as f:
        grd_names = f.readlines()
    pano_datas = [data[:-1] for data in grd_names]
    location_imgs.extend([data.split()[0] for data in pano_datas])

    random.seed(10)
    rand_ind = list(range(len(location_imgs)))
    random.shuffle(rand_ind)

    p_num = int(positive_ratio * len(location_imgs))
    positive_ind = list(range(p_num))
    negative_ind = list(range(p_num, len(location_imgs)))

    sat_dir = osp.join(raw_dir, city, 'sat_img')
    flags = []
    for i in range(p_num):
        pano_img = location_imgs[i]
        pano_info = decom_pano_name(pano_img)
        flags.append(pano_info['flag'])

    for i in range(p_num, len(location_imgs)):
        pano_img = location_imgs[i]
        pano_info = decom_pano_name(pano_img)
        lon, lat = pano_info['lon'], pano_info['lat']

        while True:
            ng_ind = random.randint(0, len(location_imgs) - 1)
            ng_img = location_imgs[ng_ind]
            ng_info = decom_pano_name(ng_img)
            ng_flag = ng_info['flag']
            tl_path = osp.join(sat_dir, ng_flag, 'tl_pos.txt')
            ref_range = np.loadtxt(tl_path, delimiter=' ')
            tl_lon, tl_lat = ref_range[0]
            br_lon, br_lat = ref_range[1]
            is_in_range = (lon < br_lon) & (lon > tl_lon) & (lat < tl_lat) & (lat > br_lat)
            if is_in_range:
                print('find a fail instance')
            else:
                break
        flags.append(ng_flag)

    with open(nfa_valid_path, "w+") as file:
        for flag in flags:
            file.write(flag + "\n")


# 临时函数
def test():
    txt_path = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing/Chicago/train.txt'
    with open(txt_path, 'r') as f:
        grd_datas = f.readlines()

    slice_ips = []
    for line in grd_datas:
        data = line.split(' ')
        slice_ips.append([float(data[1]), float(data[2])])
        pass
    slice_ips = np.array(slice_ips)
    c_ip = np.array([640, 640])
    dist = np.sqrt((slice_ips[:, 0] - c_ip[0]) ** 2 + (slice_ips[:, 1] - c_ip[1]) ** 2)
    mean_dist = np.mean(dist)
    pass


def slice_pano_image(data):
    file, o_rand = data
    city = _global_city
    raw_dir = _global_raw_dir
    output_dir = _global_output_dir
    angle_range = _ori_range
    slice_num = _slice_num

    slice_flag = 'slice'
    pano_dir = osp.join(raw_dir, city, 'panorama')
    depth_dir = osp.join(raw_dir, city, 'depth', 'North')
    sat_dir = osp.join(raw_dir, city, 'sat_img')
    output_file = osp.join(output_dir, city)
    slice_file = osp.join(output_file, slice_flag)

    resolution_x = 512
    resolution_y = 512
    fov = 90.0
    phi = 30.0  # 与水平方向夹角为向下看phi°
    default_dist = 80  # 默认为80个像素
    crop_img = True

    file_path = os.path.join(pano_dir, file)
    img_info = decom_pano_name(file_path)
    depth_path = osp.join(depth_dir, img_info['i_name'] + '.depthmap.jpg')

    p_path = osp.join(slice_file, img_info['flag'])
    create_file(p_path)
    coor_path = osp.join(slice_file, img_info['flag'], 'coordinate.npy')
    if os.path.exists(coor_path):
        return

    # 创建坐标转换的类
    sat_path = osp.join(sat_dir, img_info['flag'], 'satellite.jpg')
    sat_size = cv2.imread(sat_path).shape[0]
    tl_path = osp.join(sat_dir, img_info['flag'], 'tl_pos.txt')
    i_trans = GeoTrans(json_path=None, tl_path=tl_path, sat_size=sat_size)
    sat_img = cv2.imread(sat_path)
    sat_img = cv2.circle(sat_img, center=(int(sat_img.shape[1] / 2), int(sat_img.shape[0] / 2)), radius=3,
                         thickness=1,
                         color=(255, 0, 0))

    img = cv2.imread(file_path)
    depth = cv2.imread(depth_path).astype('float64')
    coor_list = []

    # 旋转的角度的随机值
    rand_angle = o_rand * angle_range
    if crop_img:
        for i in range(slice_num):
            theta = i * 360 / slice_num + rand_angle
            out_img, map_w1, map_h1 = crop_panorama_image(img, theta=theta, phi=phi, res_x=resolution_x, \
                                                          res_y=resolution_y, fov=fov, debug=False, output_ind=True)
            map_xy = np.dstack((map_w1, map_h1))
            sat_ips = find_coor(depth, map_xy, i_trans, img_info)
            if sat_ips is None:
                # 计算视觉地标空间距离失败，使用默认的距离
                cx, cy = sat_img.shape[1] / 2, sat_img.shape[0] / 2
                sat_ips = rotate_point(cx, cy + default_dist, cx, cy, -rand_angle - theta)
                sat_ips = np.array(sat_ips)[None]
                print('use default distance:', img_info['flag'])

            mv = np.mean(sat_ips, axis=0)
            sat_img = cv2.circle(sat_img, center=(int(mv[0]), int(mv[1])), radius=3, thickness=1,
                                 color=(0, 0, 255))
            coor_list.append(mv)
            output_file_path = os.path.join(p_path, str(i) + '.jpg')

            # 保存提取出的图像
            cv2.imwrite(output_file_path, out_img)
        # temp_path = osp.join(p_path, 'sat_res.jpg') # 保存定位在卫星上的投影
        # cv2.imwrite(temp_path, sat_img)
        coor_mat = np.array(coor_list)
        np.save(coor_path, coor_mat)


def slice_pano_image_vigor(data):
    file = data
    city = _global_city
    raw_dir = _global_raw_dir
    output_dir = _global_output_dir

    slice_flag = 'slice'
    pano_dir = osp.join(raw_dir, city, 'panorama')
    output_file = osp.join(output_dir, city)
    slice_file = osp.join(output_file, slice_flag)

    resolution_x = 512
    resolution_y = 512
    fov = 90.0
    phi = 30.0
    slice_num = 12
    crop_img = True

    file_path = os.path.join(pano_dir, file)
    img_info = decom_pano_name_vigor(file_path)
    p_path = osp.join(slice_file, img_info['flag'])
    create_file(p_path)

    img = cv2.imread(file_path)
    coor_list = []
    # 旋转的角度的随机值
    if crop_img:
        for i in range(slice_num):
            theta = i * 360 / slice_num
            out_img, map_w1, map_h1 = crop_panorama_image(img, theta=theta, phi=phi, res_x=resolution_x, \
                                                          res_y=resolution_y, fov=fov, debug=False, output_ind=True)
            output_file_path = os.path.join(p_path, str(i) + '.jpg')
            # 保存提取出的图像
            cv2.imwrite(output_file_path, out_img)


# 对每个城市生成相关的数据
def generate_city_data(city, raw_dir, output_dir, ori_range=20, crop_img=True, slice_num=12):
    print('processing city:', city)
    slice_flag = 'slice'
    pano_dir = osp.join(raw_dir, city, 'panorama')

    sat_dir = osp.join(raw_dir, city, 'sat_img')
    depth_dir = osp.join(raw_dir, city, 'depth', 'North')
    rand_mat_path = osp.join(output_dir, city, 'rand_matrix.npy')

    # 创建相关的文件夹
    output_file = osp.join(output_dir, city)
    create_file(output_file)
    slice_file = osp.join(output_file, slice_flag)
    create_file(slice_file)

    # 进行切片并保存图片、对应矩阵
    files = [f for f in os.listdir(pano_dir) if f.endswith('.jpg')]

    pano_num = len(files)
    np.random.seed(100)
    if os.path.exists(rand_mat_path):
        rand_mat = np.load(rand_mat_path)
    else:
        rand_mat = np.random.uniform(-1, 1, size=(pano_num, 3))  # 随机矩阵: 旋转，x_offset，y_offset
        np.save(rand_mat_path, rand_mat)

    # 输出文件夹，用于保存裁剪后的图像
    os.makedirs(output_dir, exist_ok=True)
    if crop_img:
        o_rands = rand_mat[:, 0].tolist()
        datas = [[file, o_rand] for file, o_rand in zip(files, o_rands)]

        num_processes = 20

        def initializer(city, raw_dir, output_dir, ori_range, slice_num):
            global _global_city, _global_raw_dir, _global_output_dir, _ori_range, _slice_num
            _global_city = city
            _global_raw_dir = raw_dir
            _global_output_dir = output_dir
            _ori_range = ori_range
            _slice_num = slice_num

        pool = Pool(processes=num_processes, initializer=initializer,
                    initargs=(city, raw_dir, output_dir, ori_range, slice_num))

        # 并行处理任务
        pool.map(slice_pano_image, datas)
        # 关闭进程池
        pool.close()
        pool.join()

    return files, rand_mat


# 多线程划分全景影像切片图像
# 对全景影像添加误差，并生成相关txt文件
def main6(argv):
    # 遍历全景并切块，生成对应的txt列表
    slice_num = 16
    default_dist = 80  # 默认为80个像素
    split_ratio = 0.5  # 80%的数据用于训练
    ori_range = 0  # 旋转噪声的范围

    # citys = ['Chicago', 'Sydney', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    citys = ['Chicago']

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    output_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_16'
    random.seed(100)

    print('start slice panorama with ori range: ', ori_range)
    print('output path: ', output_dir)
    for city in citys:
        # 对每个城市进行数据生成操作
        pano_names, rand_mat = generate_city_data(city, raw_dir, output_dir, ori_range, crop_img=True,
                                                  slice_num=slice_num)

        cross_txt = osp.join(output_dir, city, 'pano_label_balanced.txt')  # cross时使用
        same_train_txt = osp.join(output_dir, city, 'same_area_balanced_train.txt')  # same 训练时使用
        same_test_txt = osp.join(output_dir, city, 'same_area_balanced_test.txt')  # same 测试时使用

        # 生成用于定位的txt文件
        rand_ind = list(range(len(pano_names)))
        random.shuffle(rand_ind)
        half_num = int(split_ratio * len(rand_ind))
        same_train_ind = rand_ind[0:half_num]
        same_test_ind = rand_ind[half_num:]

        # pano_label_balanced.txt
        with open(cross_txt, "w+") as file:
            for i in rand_ind:
                line = f"{pano_names[i]} {rand_mat[i, 0]:.4f} {rand_mat[i, 1]:.4f} {rand_mat[i, 2]:.4f}"
                file.write(line + "\n")
        # same_area_balanced_train.txt
        with open(same_train_txt, "w+") as file:
            for i in same_train_ind:
                line = f"{pano_names[i]} {rand_mat[i, 0]:.4f} {rand_mat[i, 1]:.4f} {rand_mat[i, 2]:.4f}"
                file.write(line + "\n")
        # same_area_balanced_test.txt
        with open(same_test_txt, "w+") as file:
            for i in same_test_ind:
                line = f"{pano_names[i]} {rand_mat[i, 0]:.4f} {rand_mat[i, 1]:.4f} {rand_mat[i, 2]:.4f}"
                file.write(line + "\n")


# 根据txt中的内容生成打乱的Slice数据，用于训练Slice SkyMap
def generate_train_txt(flag_train_txt, train_txt, pro_path, slice_num=12):
    # 保存到txt中：flag, rand_x, rand_y, gtx, gty
    with open(flag_train_txt, 'r') as f:
        datas = f.readlines()

    data1 = []
    data2 = []
    for data in datas:
        parts = data.split()
        grd_name, rand_o, rand_x, rand_y = parts
        flag = decom_pano_name(grd_name)['flag']
        imgs = [[osp.join(flag, f"{i}.jpg"), rand_o, rand_x, rand_y] for i in range(slice_num)]
        data1.extend(imgs)
        coor_path = osp.join(pro_path, flag, 'coordinate.npy')
        coors = np.load(coor_path).tolist()
        data2.extend(coors)
    temp = list(range(len(data1)))
    random.shuffle(temp)
    half_temp = temp

    with open(train_txt, "w+") as file:
        for i in half_temp:
            line = f"{data1[i][0]} {data1[i][1]} {data1[i][2]} {data1[i][3]} {data2[i][0]:.3f} {data2[i][1]:.3f}"
            file.write(line + "\n")


# 根据location_test和location_train生成用于切片模型定位的训练数据
# 使用一半的数据，即训练个数为num_panoram * slice_num / 2
def main7(argv):
    slice_num = 16
    # citys = ['Chicago', 'Sydney']
    citys = ['Chicago']
    # citys = ['London']
    # citys = ['London']

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    output_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_16'
    slice_flag = 'slice'
    random.seed(100)

    for city in citys:
        cross_txt = osp.join(output_dir, city, 'pano_label_balanced.txt')
        same_test_txt = osp.join(output_dir, city, 'same_area_balanced_test.txt')
        same_train_txt = osp.join(output_dir, city, 'same_area_balanced_train.txt')

        slice_cross_txt = osp.join(output_dir, city, 'slice_cross.txt')
        slice_same_test_txt = osp.join(output_dir, city, 'slice_same_test.txt')
        slice_same_train_txt = osp.join(output_dir, city, 'slice_same_train.txt')

        pro_path = osp.join(output_dir, city, slice_flag)

        generate_train_txt(cross_txt, slice_cross_txt, pro_path, slice_num)
        generate_train_txt(same_test_txt, slice_same_test_txt, pro_path, slice_num)
        generate_train_txt(same_train_txt, slice_same_train_txt, pro_path, slice_num)


def test_zip():
    tar_path = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Test/sub_img.zip'
    extract_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Test/zip_test'

    try:
        # 使用tarfile.open打开.tar文件
        with tarfile.open(tar_path, 'r:*') as tar:
            # 'r:*'模式会自动检测压缩格式
            # 解压到指定目录
            tar.extractall(path=extract_dir)
        print(f"文件已成功解压到 {extract_dir}")
    except tarfile.TarError as e:
        print(f"解压文件时发生错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")


def increase_blue_transparency(image, transparency_factor=0.5, ratio=None):
    # 读取图像
    # 将图像从BGR转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定义蓝色的HSV范围
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    # 创建掩码，找到蓝色区域
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # 将图像从BGR转换为RGBA颜色空间
    b, g, r = cv2.split(image)
    rgba_image = cv2.merge((b, g, r, np.full(image.shape[:2], 255, dtype=np.uint8)))  # 初始透明度为255（不透明）
    # 根据掩码调整蓝色区域的透明度
    rgba_image[..., 3] = rgba_image[..., 3] * (1 - mask.astype(np.float32) * transparency_factor)
    # rgba_image[..., 3] = rgba_image[..., 3] * ratio.squeeze()
    rgba_image[..., 3] = np.clip(rgba_image[..., 3], 0, 255).astype(np.uint8)  # 确保透明度值在0-255之间
    return rgba_image


# 根据实验的结果，选取卫星影像
# 进行TP,TN,FP,FN的可视化实验，绘制slice image 的定位情况
# 绘制热力图
def main8(argv):
    # 将原卫星与现图像进行拼接
    # 现图像包含定位的12个点对
    # 从same_area_balanced_test.txt中获取进行的随机偏移
    shift_range = 160
    crop_size = 640

    min_nfa = 1

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    pro_dir = '/media/xmt/新加卷/Data/SkyMap_Process_0'
    res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Exp_Result'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Plot_Result'

    heatmap_file = 'Heatmap'
    city = 'London'
    data_txt = 'same_area_balanced_train.txt'
    sat_txt = 'half_select_sat2.txt'
    res_txt = 'result.txt'
    heat_dir = osp.join(res_dir, city, heatmap_file)

    # 加载data数据，并保存为字典，{flag: data}
    flag_dict = {}
    data_path = osp.join(pro_dir, city, data_txt)
    with open(data_path, 'r') as f:
        grd_datas = f.readlines()
    # 从txt中加载sat_flag
    sat_flag_path = osp.join(pro_dir, city, sat_txt)
    with open(sat_flag_path, 'r') as f:
        temp = f.readlines()
        sat_flags = [data[:-1] for data in temp]

    for d, s_flag in zip(grd_datas, sat_flags):
        grd_name = d.split()[0]
        info = decom_pano_name(grd_name)
        flag = info['flag']
        flag_dict[flag] = {
            'data': d[:-1],
            'sat_flag': s_flag
        }

    # 加载result.txt中的数据
    res_list = []
    res_path = osp.join(res_dir, city, res_txt)
    with open(res_path, 'r') as f:
        grd_datas = f.readlines()
    res_list.extend([data[:-1] for data in grd_datas])

    # 只处理FN和TN的数据
    heatmap_flag = 'heatmap'
    heat_save_dir = osp.join(plot_dir, city, heatmap_flag)
    create_file(heat_save_dir)
    for rd in tqdm(res_list):
        flag, cls_ty, nfa = rd.split()
        exp_flag = flag_dict[flag]['sat_flag']
        nfa = float(nfa)

        cls_dir = osp.join(plot_dir, city, cls_ty)
        create_file(cls_dir)

        # if cls_ty == res_flag: # 只处理FN和TN的数据
        # 对卫星影像进行偏移
        _, _, x_rand, y_rand = flag_dict[flag]['data'].split()
        x_rand, y_rand = shift_range * float(x_rand), shift_range * float(y_rand)

        # 加载原卫星图像
        raw_sat_path = osp.join(raw_dir, city, 'sat_img', flag, 'satellite.jpg')
        with PIL.Image.open(raw_sat_path, 'r') as SatMap:
            sat_map0 = SatMap.convert('RGB')
        sat_map0 = sat_map0.transform(
            sat_map0.size, PIL.Image.AFFINE,
            (1, 0, -x_rand,
             0, 1, -y_rand),
            resample=PIL.Image.BILINEAR)
        sat_map0 = TF.center_crop(sat_map0, crop_size)  # 偏移后进行中心采样
        sat1 = np.array(sat_map0)
        sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=3,
                          thickness=1,
                          color=(255, 0, 0))

        # 加载实验用的卫星影像
        exp_sat_path = osp.join(raw_dir, city, 'sat_img', exp_flag, 'satellite.jpg')
        with PIL.Image.open(exp_sat_path, 'r') as SatMap:
            sat_map1 = SatMap.convert('RGB')
        sat_map1 = sat_map1.transform(
            sat_map1.size, PIL.Image.AFFINE,
            (1, 0, -x_rand,
             0, 1, -y_rand),
            resample=PIL.Image.BILINEAR)
        sat_map1 = TF.center_crop(sat_map1, crop_size)  # 偏移后进行中心采样
        sat2 = np.array(sat_map1)

        # 在exp_sat上绘制loc点与end点
        mat_path = osp.join(res_dir, city, flag + '.npy')
        loc_mat = np.load(mat_path)
        for pts in loc_mat:
            lx, ly, ex, ey = pts
            ld = np.sqrt((ex - lx) ** 2 + (ey - ly) ** 2)
            dx, dy = (ex - lx) / ld, (ey - ly) / ld
            ex_, ey_ = lx + 20 * dx, ly + 20 * dy
            sat2 = cv2.line(sat2, pt1=(int(lx), int(ly)),
                            pt2=(int(ex_), int(ey_)),
                            color=(0, 255, 0), thickness=5)
            sat2 = cv2.circle(sat2, center=(int(lx), int(ly)), radius=3,
                              thickness=1,
                              color=(255, 0, 0))

        ccat_img = cv2.hconcat([sat1, sat2])
        ccat_img = cv2.cvtColor(ccat_img, cv2.COLOR_BGR2RGB)
        save_img_path = osp.join(cls_dir, flag + '.jpg')
        cv2.imwrite(save_img_path, ccat_img)  # 将原卫星图像与定位图像保留下来

        # 绘制热力图，并保存下来
        heatmap_path = osp.join(heat_dir, flag + '.npy')
        heatmap = np.load(heatmap_path)
        # 旋转并拼接，裁剪每层的热力图
        new_heatmap = []
        slice_num = heatmap.shape[0]
        for i in range(slice_num):
            heading = 90 + i * 360 / 12
            hi = heatmap[i].squeeze()
            # (c, h, w) = hi.shape
            h = hi.shape[0]
            w = hi.shape[1]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -heading, 1.0)
            rotated = cv2.warpAffine(hi, M, (w, h))
            pass

            new_heatmap.append(rotated)
        heatmap = np.array(new_heatmap)
        heatmap = np.mean(heatmap, axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = cv2.resize(heatmap.squeeze(), [sat1.shape[0], sat1.shape[1]])
        heatmap = np.tile(heatmap[None, :, :], (3, 1, 1))
        heatmap = np.transpose(heatmap, (1, 2, 0))
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        n_heatmap = increase_blue_transparency(heatmap, 0.2)
        heat_save_path = osp.join(heat_save_dir, flag + '.png')
        cv2.imwrite(heat_save_path, n_heatmap)
        # 设置半透明度（alpha值）
        # alpha = 0.7
        # # 将两个图像进行叠加
        # blended_image_sat = cv2.addWeighted(sat2, alpha, heatmap, 1 - alpha, 0)

        pass

    #     if min_nfa > nfa:
    #         min_nfa = nfa
    #         min_flag = flag
    #
    # print('min nfa:', min_nfa)
    # print('min flag:', min_flag)
    # pass


# 绘制NFA与定位误差的散点图
def plot_err_nfa(errs, nfas, save_path):
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.axvline(x=-1, color='g', linestyle='--', linewidth=2)
    errs = errs[nfas < 100]
    nfas = nfas[nfas < 100]
    ax.scatter(nfas, errs, s=5, alpha=0.5, marker='o')
    ax.grid(True, which='both', linestyle='-', linewidth='0.3', color='gray')

    # 设置主要刻度间隔
    plt.tick_params(axis='x', which='major', labelsize=20, length=5, width=2, grid_alpha=0.5,
                    grid_linewidth=0.3)
    plt.tick_params(axis='y', which='major', labelsize=20, length=5, width=2, grid_alpha=0.5,
                    grid_linewidth=0.3)
    # 设置次要刻度间隔（可选）
    plt.tick_params(axis='x', which='minor', length=2.5, width=0.5, grid_alpha=0.3, grid_linewidth=0.25)
    plt.tick_params(axis='y', which='minor', length=2.5, width=0.5, grid_alpha=0.3, grid_linewidth=0.25)
    plt.savefig(save_path)
    plt.close()


def plot_nfa_ratio(ratios, save_path):
    fig, ax = plt.subplots(figsize=(12, 9))
    y0 = ratios[:, 0] * 100
    y1 = ratios[:, 1] * 100
    x = list(range(2, 21, 2))
    ax.plot(x, y0, linestyle='--', label='log(NFA)<0', color='r', marker='o', markersize=14, linewidth=2)
    ax.plot(x, y1, linestyle='--', label='log(NFA)<-1', color='g', marker='^', markersize=14, linewidth=2)
    ax.tick_params(axis='x', which='major', labelsize=20, length=5, width=2, grid_alpha=0.5,
                   grid_linewidth=0.3)
    ax.tick_params(axis='y', which='major', labelsize=20, length=5, width=2, grid_alpha=0.5,
                   grid_linewidth=0.3)
    ax.set_xticks(x)
    ax.grid(True, which='both', linestyle='-', linewidth='0.3', color='gray')

    plt.savefig(save_path)
    plt.close()


# 绘制根据全景的定位结果，绘制频率分布图
# 处理Slice-Loc的定位结果
def main9(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = False
    NFA_thr = 0

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    # exp_res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Query_Rand'
    exp_res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/EF-Slice-Loc/CV-CoLoc'
    # exp_res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_16_Res'
    # exp_res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc_Weakly'
    exp_res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Exp_res'

    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/EF-Slice-Loc/plot_temp'
    is_cross = False
    ori_list = ['0.0']  # ['0.0', '20.0', '45.0']
    if is_cross:
        dir_label = 'cross_'
        # city_list = ['Sydney', 'Chicago', 'Johannesburg']
        city_list = ['Chicago']
    else:
        dir_label = 'same_'
        # city_list = ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
        city_list = ['Sydney']

    # txt_name = 'result_uniform_140.txt'
    txt_name = 'result.txt'

    ori = '0.0'
    dir_label += ori
    # dir_label = 'slice_cross'
    save_dir = osp.join(exp_res_dir, dir_label)
    plot_dir = osp.join(plot_dir, dir_label)
    pro_dir = '/media/xmt/新加卷1/Data/SkyMap_Process_20'

    city = 'Sydney'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    save_txt = osp.join(save_dir, city, txt_name)  # flag, lxy, err, NFA,

    slice_loc_path = osp.join(save_dir, city, 'SliceLoc')
    loc_file = 'LocRes'
    loc_dir = osp.join(plot_dir, city, loc_file)
    if not osp.exists(loc_dir):
        os.makedirs(loc_dir)

    # same_area_balanced_train.txt

    data_txt = 'pano_label_balanced.txt'
    data_path = osp.join(pro_dir, city, data_txt)
    with open(data_path, 'r') as f:
        grd_datas = f.readlines()
    flag_dict = {}
    for d in grd_datas:
        grd_name = d.split()[0]
        info = decom_pano_name(grd_name)
        flag = info['flag']
        flag_dict[flag] = {
            'data': d[:-1]
        }

    # 加载result.txt中的数据
    nfas = []
    errs = []
    oerrs = []
    in_nums = []
    with open(save_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            flag, lx, ly, err, nfa, oerr = \
                temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]

            # flag, lx, ly, err, nfa= line.split()
            lx, ly, err, nfa = float(lx), float(ly), float(err), float(nfa)
            if lx < 0 or lx > 640 or ly < 0 or ly > 640:
                continue
            if err < 8 and random.random() < 0.6:
                continue

            nfas.append(float(nfa))
            errs.append(float(err))
            oerrs.append(abs(float(oerr)))
            # in_nums.append(in_num)
    errs = np.array(errs)
    nfas = np.array(nfas)
    oerrs = np.array(oerrs)
    # in_nums = np.array(in_nums)
    save_path = osp.join(plot_dir, city, 'nfa-locerr.png')
    plot_err_nfa(errs, nfas, save_path)

    # nfa_valid = nfas < NFA_thr
    nfa_valid = np.ones(len(errs), dtype=bool)
    # nfa_valid = nfa_valid_
    # in_valid = in_nums > 3
    # nfa_valid = nfa_valid & in_valid
    print('mean:', np.mean(errs[nfa_valid]))
    print('ratio: ', np.sum(nfa_valid) / len(nfas))
    print('output slice-loc result:')
    print('all num / in num / PoR:', len(nfa_valid), np.sum(nfa_valid), round(np.sum(nfa_valid) / len(nfa_valid), 4))
    v_lerrs = errs[nfa_valid]
    print('the location result:')
    statistic_data(v_lerrs)
    v_oerrs = oerrs[nfa_valid]
    print('the orientation result:')
    statistic_data(v_oerrs)
    print('=======================================\n')

    # 绘制比例折线图
    nfa_dict = {}
    num = 21
    ratios = []
    for d in range(2, num, 2):
        s_ind = (d - 2 < errs) & (errs < d)
        errs_ = errs[s_ind]
        nafs_ = nfas[s_ind]
        r0 = np.sum(nafs_ > NFA_thr) / len(nafs_)
        rm1 = np.sum(nafs_ > NFA_thr - 1) / len(nafs_)
        nfa_dict[str(d)] = r0
        print(r0 * 100, ' ', rm1 * 100)
        ratios.append([r0, rm1])
    ratios = np.array(ratios)
    save_path = osp.join(plot_dir, city, 'nfa-ratio-line.png')
    plot_nfa_ratio(ratios, save_path)

    # 绘制定位真值，slice定位结果，
    if plot_loc:
        for line1, line2 in zip(lines, lines_ccvpe):
            flag, slx, sly, err, nfa = line1.split()
            flag2, clx, cly, err2 = line2.split()
            slx, sly, err, nfa = float(slx), float(sly), float(err), float(nfa)
            clx, cly, err2 = float(clx), float(cly), float(err2)

            _, _, x_rand, y_rand = flag_dict[flag]['data'].split()
            x_rand, y_rand = shift_range * float(x_rand), shift_range * float(y_rand)

            # err比err2 至少小5， NFA < -1
            if not ((err2 - err) > 5 and nfa < -1):
                continue

            # 加载原卫星图像
            raw_sat_path = osp.join(raw_dir, city, 'sat_img', flag, 'satellite.jpg')
            with PIL.Image.open(raw_sat_path, 'r') as SatMap:
                sat_map0 = SatMap.convert('RGB')
            sat_map0 = sat_map0.transform(
                sat_map0.size, PIL.Image.AFFINE,
                (1, 0, -x_rand,
                 0, 1, -y_rand),
                resample=PIL.Image.BILINEAR)
            sat_map0 = TF.center_crop(sat_map0, crop_size)  # 偏移后进行中心采样
            sat1 = np.array(sat_map0)
            sat0 = np.array(sat_map0)

            # 在exp_sat上绘制loc点与end点
            mat_path = osp.join(slice_loc_path, flag + '.npy')
            loc_mat = np.load(mat_path)
            # 将Slice定位的结果: x, y, ori绘制上去
            for pts in loc_mat:
                lx, ly, ex, ey = pts
                ld = np.sqrt((ex - lx) ** 2 + (ey - ly) ** 2)
                dx, dy = (ex - lx) / ld, (ey - ly) / ld
                ex_, ey_ = lx + 20 * dx, ly + 20 * dy
                sat0 = cv2.line(sat0, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 255, 0), thickness=5)
                sat0 = cv2.circle(sat0, center=(int(lx), int(ly)), radius=3,
                                  thickness=1,
                                  color=(255, 0, 0))

            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=1,
                              thickness=-1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=3,
                              thickness=1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(slx), int(sly)), radius=3,
                              thickness=-1,
                              color=(255, 0, 0))  # Slice-Loc
            sat1 = cv2.circle(sat1, center=(int(clx), int(cly)), radius=2,
                              thickness=-1,
                              color=(0, 255, 0))  # CCVPE
            sat1 = cv2.circle(sat1, center=(int(clx), int(cly)), radius=3,
                              thickness=1,
                              color=(255, 0, 255))  # CCVPE

            img_path = osp.join(loc_dir, flag + '.png')
            ccat_img = cv2.hconcat([sat0, sat1])
            ccat_img = cv2.cvtColor(ccat_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, ccat_img)


def main10(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = True

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    save_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Exp_Result'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Plot_Result'
    pro_dir = '/media/xmt/新加卷1/Data/SkyMap_Process_0'

    city = 'Chicago'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    txt_name = 'result.txt'

    select_file = 'select_pano'
    select_dir = osp.join(plot_dir, city, select_file)
    files = [f for f in os.listdir(select_dir) if f.endswith('.png')]

    # same_area_balanced_train.txt
    data_txt = 'pano_label_balanced.txt'
    data_path = osp.join(pro_dir, city, data_txt)
    new_label_path = osp.join(plot_dir, city, 'new_label.txt')
    with open(data_path, 'r') as f:
        grd_datas = f.readlines()
    flag_dict = {}
    for d in grd_datas:
        grd_name = d.split()[0]
        info = decom_pano_name(grd_name)
        flag = info['flag']
        flag_dict[flag] = d[:-1]

    with open(new_label_path, "w+") as file:
        for i_name in files:
            flag = i_name[:-4]
            file.write(flag_dict[flag] + "\n")

    # 加载选取的影像的flag
    # 加载训练txt
    # 保存成新的子txt


# 根据路径生成slice-loc的热力图
def get_slice_heatmap(heatmap_path, resize=640):
    heatmap0 = np.load(heatmap_path)
    # 旋转并拼接，裁剪每层的热力图
    new_heatmap = []
    slice_num = heatmap0.shape[0]
    a_img = np.full((512, 512), 0.)
    for i in range(slice_num):
        heading = 90 + i * 360 / 12
        hi = heatmap0[i].squeeze()
        # (c, h, w) = hi.shape
        h = hi.shape[0]
        w = hi.shape[1]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -heading, 1.0)
        rotated = cv2.warpAffine(hi, M, (w, h))
        rotated = (rotated - rotated.min()) / (rotated.max() - rotated.min())
        new_heatmap.append(rotated)
        a_img += rotated

    a_img = (a_img - a_img.min()) / (a_img.max() - a_img.min())
    a_img = cv2.resize(a_img.squeeze(), [resize, resize])
    heatmap = np.array(new_heatmap)
    heatmap = np.sum(heatmap, axis=0)
    heatmap[heatmap > 1] = 1
    heatmap = cv2.resize(heatmap.squeeze(), [resize, resize])

    heatmap = np.tile(heatmap[None, :, :], (3, 1, 1))
    heatmap = np.transpose(heatmap, (1, 2, 0))
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # n_heatmap = np.concatenate((heatmap, a_img), axis=2)

    n_heatmap = increase_blue_transparency(heatmap, 0.5, a_img)
    return n_heatmap


# 根据路径生成ccvpe的热力图
def get_ccvpe_heatmap(heatmap_path, resize=640):
    heatmap = np.load(heatmap_path)
    heatmap = np.mean(heatmap, axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = cv2.resize(heatmap.squeeze(), [resize, resize])
    a_img = heatmap.copy()

    heatmap = np.tile(heatmap[None, :, :], (3, 1, 1))
    heatmap = np.transpose(heatmap, (1, 2, 0))
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # n_heatmap = np.concatenate((heatmap, a_img), axis=2)
    n_heatmap = increase_blue_transparency(heatmap, 0.5, a_img)
    return n_heatmap


def crop_img(image):
    cx, cy = image.shape[0] / 2, image.shape[0] / 2
    lenx, leny = 200, 200
    ux, uy, dx, dy = int(cx + lenx), int(cy + leny), int(cx - lenx), int(cy - leny)
    new_img = image[dy:uy, dx:ux, :]
    return new_img


# 绘制根据全景的定位结果，绘制频率分布图
# 处理Slice-Loc的定位结果
# 生成与ccvpe的对比图
def main11(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = True

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    save_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_H0'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_H0_plot'
    pro_dir = '/media/xmt/新加卷1/Data/SkyMap_Process_0'

    city = 'Chicago'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    txt_name = 'result.txt'
    save_txt = osp.join(save_dir, city, 'result.txt')  # flag, lxy, err, NFA,
    save_txt2 = osp.join(save_dir, city, 'result_ccvpe.txt')  # flag, lxy, err, NFA,
    with open(save_txt, 'r') as f:
        lines = f.readlines()
    with open(save_txt2, 'r') as f:
        lines_ccvpe = f.readlines()

    slice_loc_path = osp.join(save_dir, city, 'SliceLoc')
    loc_file = 'LocRes'
    loc_dir = osp.join(plot_dir, city, loc_file)
    if not osp.exists(loc_dir):
        os.makedirs(loc_dir)
    heatmap_dir = osp.join(plot_dir, city, 'Heatmap')
    if not osp.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    csat_dir = osp.join(plot_dir, city, 'ConSat')
    if not osp.exists(csat_dir):
        os.makedirs(csat_dir)

    slice_heat_dir = osp.join(save_dir, city, 'Heatmap')
    ccvpe_heat_dir = osp.join(save_dir, city, 'Heatmap_ccvpe')

    # same_area_balanced_train.txt
    data_txt = 'pano_label_balanced.txt'
    data_path = osp.join(pro_dir, city, data_txt)
    with open(data_path, 'r') as f:
        grd_datas = f.readlines()
    flag_dict = {}
    for d in grd_datas:
        grd_name = d.split()[0]
        info = decom_pano_name(grd_name)
        flag = info['flag']
        flag_dict[flag] = {
            'data': d[:-1]
        }

    # 绘制定位真值，slice定位结果，
    if plot_loc:
        for line1, line2 in zip(lines, lines_ccvpe):
            flag, slx, sly, err, nfa = line1.split()
            flag2, clx, cly, err2 = line2.split()
            slx, sly, err, nfa = float(slx), float(sly), float(err), float(nfa)
            clx, cly, err2 = float(clx), float(cly), float(err2)

            _, _, x_rand, y_rand = flag_dict[flag]['data'].split()
            x_rand, y_rand = shift_range * float(x_rand), shift_range * float(y_rand)

            # err比err2 至少小5， NFA < -1
            if not ((err2 - err) > 3 and nfa < 0):
                continue

            # 加载原卫星图像
            raw_sat_path = osp.join(raw_dir, city, 'sat_img', flag, 'satellite.jpg')
            with PIL.Image.open(raw_sat_path, 'r') as SatMap:
                sat_map0 = SatMap.convert('RGB')
            sat_map0 = sat_map0.transform(
                sat_map0.size, PIL.Image.AFFINE,
                (1, 0, -x_rand,
                 0, 1, -y_rand),
                resample=PIL.Image.BILINEAR)
            sat_map0 = TF.center_crop(sat_map0, crop_size)  # 偏移后进行中心采样
            sat1 = np.array(sat_map0)
            sat0 = np.array(sat_map0)

            # 在exp_sat上绘制loc点与end点
            mat_path = osp.join(slice_loc_path, flag + '.npy')
            loc_mat = np.load(mat_path)
            # 将Slice定位的结果: x, y, ori绘制上去
            for pts in loc_mat:
                lx, ly, ex, ey = pts
                ld = np.sqrt((ex - lx) ** 2 + (ey - ly) ** 2)
                dx, dy = (ex - lx) / ld, (ey - ly) / ld
                ex_, ey_ = lx + 15 * dx, ly + 15 * dy
                sat0 = cv2.line(sat0, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 255, 0), thickness=3)
                sat0 = cv2.circle(sat0, center=(int(lx), int(ly)), radius=3,
                                  thickness=1,
                                  color=(255, 0, 0))

            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=1,
                              thickness=-1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=3,
                              thickness=1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(slx), int(sly)), radius=3,
                              thickness=-1,
                              color=(255, 0, 0))  # Slice-Loc
            sat1 = cv2.circle(sat1, center=(int(clx), int(cly)), radius=2,
                              thickness=-1,
                              color=(0, 255, 0))  # CCVPE
            sat1 = cv2.circle(sat1, center=(int(clx), int(cly)), radius=3,
                              thickness=1,
                              color=(255, 0, 255))  # CCVPE

            img_path = osp.join(loc_dir, flag + '.png')
            s0_, s1_ = crop_img(sat0), crop_img(sat1)
            ccat_img = cv2.hconcat([s0_, s1_])
            ccat_img = cv2.cvtColor(ccat_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, ccat_img)

            # 加载热力图数据并绘制
            s_heatmap_path = osp.join(slice_heat_dir, flag + '.npy')
            s_heatmap = get_slice_heatmap(s_heatmap_path)

            # c_heatmap_path = osp.join(ccvpe_heat_dir, flag+'.npy')
            # c_heatmap = get_ccvpe_heatmap(c_heatmap_path)
            # s_heatmap, c_heatmap = crop_img(s_heatmap), crop_img(c_heatmap)
            # ccat_img = cv2.hconcat([s_heatmap, c_heatmap])
            # # ccat_img = cv2.cvtColor(ccat_img, cv2.COLOR_BGR2RGB)
            # img_path = osp.join(heatmap_dir, flag+'.png')
            # cv2.imwrite(img_path, ccat_img)

            # 绘制热力图叠加影像
            sat0_ = cv2.circle(sat0, center=(int(slx), int(sly)), radius=3,
                               thickness=-1,
                               color=(255, 0, 0))  # Slice-Loc 设置为实心原点
            sat0_ = cv2.circle(sat0_, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)),
                               radius=3,
                               thickness=1,
                               color=(0, 0, 255))  # 真值 设置为空心
            sat1 = np.array(sat_map0)
            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=1,
                              thickness=-1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=3,
                              thickness=1,
                              color=(0, 0, 255))  # 真值
            sat1_ = cv2.circle(sat1, center=(int(clx), int(cly)), radius=3,
                               thickness=-1,
                               color=(255, 0, 255))  # CCVPE
            sat0_, sat1_ = crop_img(sat0_), crop_img(sat1_)
            sat_imgs = cv2.hconcat([sat0_, sat1_])
            sat_imgs = cv2.cvtColor(sat_imgs, cv2.COLOR_BGR2RGB)
            sats_save_path = osp.join(csat_dir, flag + '.png')
            cv2.imwrite(sats_save_path, sat_imgs)

            pass


# 查看误差较小，单是NFA>0 的原因
def main12(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = True

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    save_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Exp_Result_new'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Err_nfa'
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_45'

    city = 'Chicago'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    txt_name = 'result.txt'
    save_txt = osp.join(save_dir, city, 'result.txt')  # flag, lxy, err, NFA,
    save_txt2 = osp.join(save_dir, city, 'result_ccvpe.txt')  # flag, lxy, err, NFA,
    with open(save_txt, 'r') as f:
        lines = f.readlines()
    with open(save_txt2, 'r') as f:
        lines_ccvpe = f.readlines()

    slice_loc_path = osp.join(save_dir, city, 'SliceLoc')
    loc_file = 'LocRes'
    loc_dir = osp.join(plot_dir, city, loc_file)
    if not osp.exists(loc_dir):
        os.makedirs(loc_dir)
    heatmap_dir = osp.join(plot_dir, city, 'Heatmap')
    if not osp.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    csat_dir = osp.join(plot_dir, city, 'err_NFA')
    if not osp.exists(csat_dir):
        os.makedirs(csat_dir)

    # same_area_balanced_train.txt
    data_txt = 'pano_label_balanced.txt'
    data_path = osp.join(pro_dir, city, data_txt)
    with open(data_path, 'r') as f:
        grd_datas = f.readlines()
    flag_dict = {}
    for d in grd_datas:
        grd_name = d.split()[0]
        info = decom_pano_name(grd_name)
        flag = info['flag']
        flag_dict[flag] = {
            'data': d[:-1]
        }

    # 绘制定位真值，slice定位结果，
    if plot_loc:
        for line1 in lines:
            flag, slx, sly, err, nfa = line1.split()
            slx, sly, err, nfa = float(slx), float(sly), float(err), float(nfa)

            _, _, x_rand, y_rand = flag_dict[flag]['data'].split()
            x_rand, y_rand = shift_range * float(x_rand), shift_range * float(y_rand)

            # err比err2 至少小5， NFA < -1
            if not (nfa > 0 and err < 3):
                continue

            # 加载原卫星图像
            raw_sat_path = osp.join(raw_dir, city, 'sat_img', flag, 'satellite.jpg')
            with PIL.Image.open(raw_sat_path, 'r') as SatMap:
                sat_map0 = SatMap.convert('RGB')
            sat_map0 = sat_map0.transform(
                sat_map0.size, PIL.Image.AFFINE,
                (1, 0, -x_rand,
                 0, 1, -y_rand),
                resample=PIL.Image.BILINEAR)
            sat_map0 = TF.center_crop(sat_map0, crop_size)  # 偏移后进行中心采样
            sat1 = np.array(sat_map0)
            sat0 = np.array(sat_map0)

            # 在exp_sat上绘制loc点与end点
            mat_path = osp.join(slice_loc_path, flag + '.npy')
            loc_mat = np.load(mat_path)
            # 将Slice定位的结果: x, y, ori绘制上去
            for pts in loc_mat:
                lx, ly, ex, ey = pts
                ld = np.sqrt((ex - lx) ** 2 + (ey - ly) ** 2)
                dx, dy = (ex - lx) / ld, (ey - ly) / ld
                ex_, ey_ = lx + 15 * dx, ly + 15 * dy
                sat0 = cv2.line(sat0, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 255, 0), thickness=3)
                sat0 = cv2.circle(sat0, center=(int(lx), int(ly)), radius=3,
                                  thickness=1,
                                  color=(255, 0, 0))

            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=1,
                              thickness=-1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=3,
                              thickness=1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(slx), int(sly)), radius=3,
                              thickness=-1,
                              color=(255, 0, 0))  # Slice-Loc

            img_path = osp.join(loc_dir, flag + '.png')
            s0_, s1_ = crop_img(sat0), crop_img(sat1)
            ccat_img = cv2.hconcat([s0_, s1_])
            ccat_img = cv2.cvtColor(ccat_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, ccat_img)

            # 绘制热力图叠加影像
            sat0_ = cv2.circle(sat0, center=(int(slx), int(sly)), radius=3,
                               thickness=-1,
                               color=(255, 0, 0))  # Slice-Loc 设置为实心原点
            sat0_ = cv2.circle(sat0_, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)),
                               radius=3,
                               thickness=1,
                               color=(0, 0, 255))  # 真值 设置为空心
            sat1 = np.array(sat_map0)
            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=1,
                              thickness=-1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=3,
                              thickness=1,
                              color=(0, 0, 255))  # 真值
            sat_imgs = sat0_
            sat_imgs = cv2.cvtColor(sat_imgs, cv2.COLOR_BGR2RGB)
            sats_save_path = osp.join(csat_dir, flag + '.png')
            cv2.imwrite(sats_save_path, sat_imgs)

            pass


# 绘制slice image 的定位结果
def main13(argv):
    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    save_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SliceLoc_Res'
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_45'

    in_datas = []
    out_datas = []
    city = 'Sydney'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    npy_dir = osp.join(save_dir, city, 'SliceLoc')
    files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    for file in files:
        data = np.load(osp.join(npy_dir, file))
        in_ = data[data[:, 6] == 1., 4:6]
        out_ = data[data[:, 6] == 0., 4:6]
        if len(in_) > 0:
            in_datas.append(in_)
            x1 = in_[:, 0]
            y1 = in_[:, 1]
            if np.sum((y1 > 70) & (x1 < 15) & (x1 > 1)) > 0:
                print(file)
        if len(out_) > 0:
            out_datas.append(out_)

    in_datas = np.vstack(in_datas)
    out_datas = np.vstack(out_datas)

    np.savetxt(osp.join(save_dir, city, 'inlier_slice.txt'), in_datas, fmt='%.3f', delimiter=' ')
    np.savetxt(osp.join(save_dir, city, 'out_slice.txt'), out_datas, fmt='%.3f', delimiter=' ')


# 绘制选中的子图定位结果
# 底图，预测值，真值
def main14(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = True

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    save_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Theta_criterion'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Theta_criterion_plot'
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_45'

    city = 'Chicago'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    txt_name = 'result.txt'
    save_txt = osp.join(save_dir, city, 'result.txt')  # flag, lxy, err, NFA,
    with open(save_txt, 'r') as f:
        lines = f.readlines()

    slice_loc_path = osp.join(save_dir, city, 'SliceLoc')
    loc_file = 'yellow'
    loc_dir = osp.join(plot_dir, city, loc_file)
    if not osp.exists(loc_dir):
        os.makedirs(loc_dir)

    # same_area_balanced_train.txt
    data_txt = 'pano_label_balanced.txt'
    data_path = osp.join(pro_dir, city, data_txt)
    with open(data_path, 'r') as f:
        grd_datas = f.readlines()
    flag_dict = {}
    for d in grd_datas:
        grd_name = d.split()[0]
        info = decom_pano_name(grd_name)
        flag = info['flag']
        flag_dict[flag] = {
            'data': d[:-1]
        }

    # 绘制定位真值，slice定位结果，
    if plot_loc:
        for line1 in lines:
            flag, slx, sly, err, nfa = line1.split()
            slx, sly, err, nfa = float(slx), float(sly), float(err), float(nfa)

            _, _, x_rand, y_rand = flag_dict[flag]['data'].split()
            x_rand, y_rand = shift_range * float(x_rand), shift_range * float(y_rand)

            # 加载原卫星图像
            raw_sat_path = osp.join(raw_dir, city, 'sat_img', flag, 'satellite.jpg')
            with PIL.Image.open(raw_sat_path, 'r') as SatMap:
                sat_map0 = SatMap.convert('RGB')
            sat_map0 = sat_map0.transform(
                sat_map0.size, PIL.Image.AFFINE,
                (1, 0, -x_rand,
                 0, 1, -y_rand),
                resample=PIL.Image.BILINEAR)
            sat_map0 = TF.center_crop(sat_map0, crop_size)  # 偏移后进行中心采样
            sat1 = np.array(sat_map0)

            # 在exp_sat上绘制loc点与end点
            mat_path = osp.join(slice_loc_path, flag + '.npy')
            loc_mat = np.load(mat_path)
            # 判断是否有需要的slice-loc
            valids = loc_mat[:, 6]
            dist1s = loc_mat[:, 4]  # x轴
            dist2s = loc_mat[:, 5]  # y轴
            is_yellow = (valids == 0.) & (dist1s < 50) & (dist1s > 10) & (dist2s > 150) & (dist2s < 190)  # yellow
            # is_yellow = (valids==0.)&(dist2s<50)&(dist2s>10)&(dist1s>150)&(dist1s<190) # green
            # is_yellow = (valids==0.)&(dist1s<15)&(dist1s>3)&(dist2s>75)&(dist2s<95) # black

            if np.sum(is_yellow) < 1:
                continue

            # 将Slice定位的结果: x, y, ori绘制上去
            qq = 0
            for kk, pts in enumerate(loc_mat):
                if not is_yellow[kk]:
                    continue
                if qq > 0:
                    continue
                qq += 1
                lx, ly, ex, ey, dist1, dist2, valid = pts
                ld = np.sqrt((ex - lx) ** 2 + (ey - ly) ** 2)
                dx, dy = (ex - lx) / ld, (ey - ly) / ld
                ex_, ey_ = lx + 40 * dx, ly + 40 * dy
                sat1 = cv2.line(sat1, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 255, 0), thickness=12)
                sat1 = cv2.circle(sat1, center=(int(lx), int(ly)), radius=14,
                                  thickness=-1,
                                  color=(255, 0, 0))

            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=1,
                              thickness=-1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=3,
                              thickness=1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(slx), int(sly)), radius=3,
                              thickness=-1,
                              color=(255, 0, 0))  # Slice-Loc

            img_path = osp.join(loc_dir, flag + '.png')
            sat1 = cv2.cvtColor(sat1, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, sat1)


# 从slice中获取切片
def main15(argv):
    path1 = '/media/xmt/563A16213A15FEA5/XMT/Datas/SliceLoc_Res_plot/Sydney/yellow'
    files = [f for f in os.listdir(path1) if f.endswith('.png')]
    path2 = '/media/xmt/563A16213A15FEA5/XMT/Datas/SliceLoc_Res_plot/Sydney/yellow_slice'
    path3 = '/media/xmt/新加卷1/Data/SkyMap_Process_0/Sydney/slice'
    for file in files:
        f_name = file[:-4]
        src_path = osp.join(path3, f_name)
        dst_path = osp.join(path2, f_name)
        shutil.copytree(src_path, dst_path)


def plot_matches(image0, image1, kpts0, kpts1, matches, margin=10, inliers=None):
    image0 = image0[:, :, ::-1]  # 将BGR转为RGB
    image1 = image1[:, :, ::-1]

    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]

    h = max(h0, h1)
    w = w0 + w1 + margin
    if len(image0.shape) == 2:
        match_img = np.zeros((h, w), np.uint8)
    else:
        match_img = np.zeros((h, w, 3), np.uint8)
    match_img[:h0, :w0] = image0
    match_img[:h1, w0 + margin:] = image1
    if len(match_img.shape) == 2:
        match_img = cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR)

    # for i in range(kpts0.shape[0]):
    #     pt = kpts0[i]
    #     match_img = cv2.circle(match_img, center=(int(pt[0]), int(pt[1])), radius=3, thickness=1, color=(0, 0, 255))
    #
    # for i in range(kpts1.shape[0]):
    #     pt = kpts1[i]
    #     match_img = cv2.circle(match_img, center=(int(pt[0] + w0 + margin), int(pt[1])), radius=3, thickness=1,
    #                            color=(0, 0, 255))

    if inliers is None:
        line_color = (0, 255, 0)  # 默认使用绿色
    else:
        line_color = (0, 0, 255)

    for i in range(matches.shape[0]):
        p0 = kpts0[matches[i, 0]]
        p1 = kpts1[matches[i, 1]]
        match_img = cv2.line(match_img, pt1=(int(p0[0]), int(p0[1])), pt2=(int(p1[0] + w0 + margin), int(p1[1])),
                             color=line_color, thickness=2)

    if inliers is not None:
        for i in inliers:
            p0 = kpts0[matches[i, 0]]
            p1 = kpts1[matches[i, 1]]
            match_img = cv2.line(match_img, pt1=(int(p0[0]), int(p0[1])), pt2=(int(p1[0] + w0 + margin), int(p1[1])),
                                 color=(0, 255, 0), thickness=2)

    # 将BGR转为RGB
    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    return match_img


def main16(argv):
    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    output_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Temp'

    city = 'Chicago'
    mch_res = 'match'
    depth_dir = osp.join(raw_dir, city, 'depth', 'North')
    pano_dir = osp.join(raw_dir, city, 'panorama')
    sat_dir = osp.join(raw_dir, city, 'sat_img')

    # 创建相关的文件夹
    output_file = osp.join(output_dir, city)
    create_file(output_file)
    mch_file = osp.join(output_file, mch_res)
    create_file(mch_file)

    file = 'CHG,8YFIUXdNGtKhOl0uZbOE-g,41.80964257,-87.74330069,183,2023.jpg'
    file_path = os.path.join(pano_dir, file)
    img_info = decom_pano_name(file_path)
    depth_path = osp.join(depth_dir, img_info['i_name'] + '.depthmap.jpg')
    p_path = osp.join(mch_file, img_info['flag'])
    create_file(p_path)

    # 加载卫星
    sat_path = osp.join(sat_dir, img_info['flag'], 'satellite.jpg')
    sat_size = cv2.imread(sat_path).shape[0]
    tl_path = osp.join(sat_dir, img_info['flag'], 'tl_pos.txt')
    i_trans = GeoTrans(json_path=None, tl_path=tl_path, sat_size=sat_size)
    sat_img = cv2.imread(sat_path)

    # 加载全景数据
    img = cv2.imread(file_path)
    depth = cv2.imread(depth_path).astype('float64')
    coor_list = []

    slice_num = 12
    resolution_x = 512
    resolution_y = 512
    fov = 90.0
    phi = 0.0
    for i in range(slice_num):
        theta = i * 360 / slice_num
        out_img, map_w1, map_h1 = crop_panorama_image(img, theta=theta, phi=phi, res_x=resolution_x, \
                                                      res_y=resolution_y, fov=fov, debug=False, output_ind=True)
        map_xy = np.dstack((map_w1, map_h1))
        sat_ips, grd_ips = find_coor2(depth, map_xy, i_trans, img_info)
        used_ind = random.sample(range(grd_ips.shape[0]), 100)
        kpt1, kpt2 = grd_ips[used_ind], sat_ips[used_ind]
        temp1 = np.arange(kpt1.shape[0])
        matches = np.vstack((temp1, temp1)).transpose()
        m_res = plot_matches(out_img, sat_img, kpt1, kpt2, matches=matches)
        output_file_path = os.path.join(p_path, str(i) + '.jpg')
        cv2.imwrite(output_file_path, m_res)

        pass
        # sat_img = cv2.circle(sat_img, center=(int(mv[0]), int(mv[1])), radius=3, thickness=1,
        #                      color=(0, 0, 255))

    # 投影


def cal_dist():
    lon_lat1 = np.array([])
    lon_lat2 = np.array([])
    pass


# 绘制噪声查询的散点结果
def main17(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = False

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    save_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_H0_0'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_H0_0_plot'
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_20'

    city = 'Chicago'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    save_txt = osp.join(save_dir, city, 'result.txt')  # flag, lxy, err, NFA,
    if not osp.exists(plot_dir):
        os.makedirs(plot_dir)
    # 加载result.txt中的数据
    # f"{x_err} {y_err} {ori_err} {sin_err} {theta_err}"
    data_mat = np.loadtxt(save_txt)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(data_mat[:, 0], data_mat[:, 2], s=5, alpha=0.5, marker='o')
    s_path1 = osp.join(plot_dir, 'x_ori.png')  # x 与 旋转误差 散点图
    plt.savefig(s_path1)
    s_path2 = osp.join(plot_dir, 'theta_ori.png')  # theta - ori 散点图
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(data_mat[:, 4], data_mat[:, 2], s=5, alpha=0.5, marker='o')
    plt.savefig(s_path2)


# 统计定位定向结果
def statistic_data(data):
    m_res = f"{round(np.mean(data), 2):.2f} {round(np.median(data), 2):.2f}"
    p_res = f"{round(np.sum(data < 1) / len(data) * 100, 2):.2f} \
    {round(np.sum(data < 3) / len(data) * 100, 2):.2f} \
    {round(np.sum(data < 5) / len(data) * 100, 2):.2f} \
    {round(np.sum(data < 8) / len(data) * 100, 2):.2f} \
    {round(np.sum(data < 10) / len(data) * 100, 2):.2f}"
    print('mean, median: ', m_res)
    print('percentage of 1, 3, 5, 8, 10: ', p_res)
    return m_res, p_res


# 单位为像素
def statistic_data_pixel(data):
    m_res = f"{round(np.mean(data), 2):.2f} {round(np.median(data), 2):.2f}"
    p_res = f"{round(np.sum(data < 10) / len(data) * 100, 2):.2f} \
    {round(np.sum(data < 20) / len(data) * 100, 2):.2f} \
    {round(np.sum(data < 30) / len(data) * 100, 2):.2f} \
    {round(np.sum(data < 50) / len(data) * 100, 2):.2f} \
    {round(np.sum(data < 80) / len(data) * 100, 2):.2f}"
    print('mean, median: ', m_res)
    print('percentage of 10, 20, 30, 50, 80: ', p_res)
    return m_res, p_res


# 统计实验结果
def main18(argv):
    shift_range = 160
    crop_size = 640
    NFA_thr = 0
    use_nfa = False

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    exp_res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Exp_res'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Plot_res'
    compare_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Compare_res'
    is_cross = True
    ori_list = ['0.0']  # ['0.0', '20.0', '45.0']
    if is_cross:
        dir_label_ = 'cross_'
        # city_list = ['Sydney']
        city_list = ['Sydney', 'Johannesburg', 'Chicago']
    else:
        dir_label_ = 'same_'
        city_list = ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
        # city_list = ['Chicago']

    txt_name = 'result.txt'
    txt_ccvpe = 'result_ccvpe.txt'
    txt_hcnet = 'result_hcnet.txt'

    for ori in ori_list:
        dir_label = dir_label_ + ori

        res_dir = osp.join(exp_res_dir, dir_label)
        if not osp.exists(res_dir):
            continue
        print('start processing ', dir_label)
        mat_sliceloc = []
        mat_ccvpe = []

        # 读取文件中每个城市的结果并统计
        for city in city_list:
            data_path = osp.join(res_dir, city, txt_name)
            # 加载result.txt中的数据
            with open(data_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    temp = line.split()
                    flag, lx, ly, lerr, nfa, oerr = \
                        temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]
                    avg_err = float(temp[6])
                    # flag, lx, ly, err, nfa= line.split()
                    lx, ly, lerr, nfa, oerr = float(lx), float(ly), float(lerr), float(nfa), abs(float(oerr))
                    if lx > 0 and lx < 640 and ly > 0 and ly < 640:
                        mat_sliceloc.append([lx, ly, lerr, nfa, oerr, avg_err])

            # 加载result_ccvpe.txt中的数据
            data_path = osp.join(res_dir, city, txt_ccvpe)
            with open(data_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # flag, lx, ly, lerr, oerr, score
                    flag, lx, ly, lerr, oerr, score = line.split()
                    lx, ly, lerr, oerr, score = float(lx), float(ly), float(lerr), abs(float(oerr)), float(score)
                    mat_ccvpe.append([lx, ly, lerr, oerr, score])

        mat_sliceloc = np.array(mat_sliceloc)
        if use_nfa:
            nfa_valid = mat_sliceloc[:, 3] < NFA_thr
        else:
            nfa_valid = np.ones(len(mat_sliceloc), dtype=bool)
        RoP = round(np.sum(nfa_valid) / len(nfa_valid), 4) * 100
        print('all num / in num / PoR:', len(nfa_valid), np.sum(nfa_valid), RoP)

        m_res = []
        p_res = []
        RoTN_res = []

        print('output hc-net result:')
        hcnet_txt = osp.join(res_dir, txt_hcnet)
        mat_hcnet = np.loadtxt(hcnet_txt)
        mat_hcnet = mat_hcnet[mat_hcnet[:, 2].argsort()[::-1]]
        if use_nfa:
            keep_size = np.sum(nfa_valid)
        else:
            keep_size = len(mat_hcnet)
        loc_errs = mat_hcnet[:keep_size, 0]
        print('the location result:')
        lm1, lp1 = statistic_data(loc_errs)  # mean, percentage
        ori_errs = mat_hcnet[:keep_size, 1]
        print('the orientation result:')
        om1, op1 = statistic_data(ori_errs)
        m_res.append(lm1 + ' ' + om1)  # mean, percentage
        p_res.append(lp1 + ' ' + op1)
        pd_valid1 = np.zeros(len(mat_hcnet), dtype=bool)
        pd_valid1[:keep_size] = True
        gt_valid1 = mat_hcnet[:, 0] < 10
        RoTN10 = statistics_valid_res(pd_valid1, gt_valid1)
        RoTN_res.append(RoTN10)
        print('=======================================\n')

        print('output ccvpe result:')
        mat_ccvpe = np.array(mat_ccvpe)
        mat_ccvpe = mat_ccvpe[mat_ccvpe[:, 4].argsort()[::-1]]
        loc_errs = mat_ccvpe[:keep_size, 2]
        print('the location result:')
        lm2, lp2 = statistic_data(loc_errs)
        ori_errs = mat_ccvpe[:keep_size, 3]
        print('the orientation result:')
        om2, op2 = statistic_data(ori_errs)
        m_res.append(lm2 + ' ' + om2)  # mean, percentage
        p_res.append(lp2 + ' ' + op2)
        pd_valid2 = np.zeros(len(mat_ccvpe), dtype=bool)
        pd_valid2[:keep_size] = True
        gt_valid2 = mat_ccvpe[:, 2] < 10
        RoTN10 = statistics_valid_res(pd_valid2, gt_valid2)
        RoTN_res.append(RoTN10)
        print('=======================================\n')

        print('output slice-loc result:')
        v_lerrs = mat_sliceloc[nfa_valid, 2]
        v_oerrs = mat_sliceloc[nfa_valid, 4]
        print('the location result:')
        lm3, lp3 = statistic_data(v_lerrs)
        print('the orientation result:')
        om3, op3 = statistic_data(v_oerrs)
        m_res.append(lm3 + ' ' + om3)  # mean, percentage
        p_res.append(lp3 + ' ' + op3)
        prd_valids = nfa_valid
        gt_valids = mat_sliceloc[:, 2] < 10
        RoTN10 = statistics_valid_res(prd_valids, gt_valids)
        RoTN_res.append(RoTN10)
        print('=======================================\n')

        avg_errs = mat_sliceloc[nfa_valid, 5]
        print('the average location result:')
        statistic_data(avg_errs)

        compare_txt = osp.join(compare_dir, dir_label + '.txt')
        with open(compare_txt, "w+") as file:
            file.write(str(RoP) + '\n')
            for data in RoTN_res:
                file.write(data + "\n")
            for data in m_res:
                file.write(data + "\n")
            for data in p_res:
                file.write(data + "\n")

        # 比较小于10米的找回率，精度等指标


def analyse_vigor():
    txt_path = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc_vigor/result_vigor.txt'
    data = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        data.append(float(line.split()[0]))
    loc_err2 = np.array(data)
    print('percentage of samples with localization error under 1m, 3m, 5m, 8m, 10m: ',
          np.sum(loc_err2 < 1) / len(loc_err2),
          np.sum(loc_err2 < 3) / len(loc_err2),
          np.sum(loc_err2 < 5) / len(loc_err2),
          np.sum(loc_err2 < 8) / len(loc_err2),
          np.sum(loc_err2 < 10) / len(loc_err2))


# 绘制三维的频率分布图
def main19(argv):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 生成一些示例数据
    np.random.seed(0)
    data_x = np.random.randn(1000)
    data_y = np.random.randn(1000)

    # 计算直方图的频率
    hist, xedges, yedges = np.histogram2d(data_x, data_y, bins=50)
    # ...（前面的代码与上面相同，直到获取到hist, xedges, yedges）
    # 获取直方图的x和y中心位置

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # 使用网格数据对直方图进行插值（可选，但通常不推荐用于直方图）
    from scipy.interpolate import griddata

    xi, yi = np.linspace(xedges[0], xedges[-1], 100), np.linspace(yedges[0], yedges[-1], 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((xcenters.ravel(), ycenters.ravel()), hist.ravel(), (xi, yi), method='cubic')

    # 绘制三维曲面
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xi, yi, zi, cmap='Blues', edgecolor='none')

    # 设置轴标签和范围（与上面相同）
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Frequency')
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])

    # 显示图形
    plt.show()


# 生成用于测试NFA有效性的txt列表文件
# 生成背景模型测试文件，从现有场景中随机选取12个切片，进行定位
def main20(argv):
    slice_num = 12
    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    pro_dir = '/media/xmt/新加卷1/Data/SkyMap_Process_0'
    city = 'Sydney'
    slice_flag = 'slice'
    pano_flag = 'panorama'

    location_path = osp.join(pro_dir, city, 'pano_label_balanced.txt')
    nfa_valid_path = osp.join(pro_dir, city, 'null_slice_test.txt')
    rand_slice_path = osp.join(pro_dir, city, 'null_slice_single.txt')  # 单个地面查询的随机flag生成
    sat_dir = osp.join(raw_dir, city, 'sat_img')

    location_imgs = []
    with open(location_path, 'r') as f:
        grd_names = f.readlines()
    pano_datas = [data[:-1] for data in grd_names]
    location_imgs.extend([data.split()[0] for data in pano_datas])

    random.seed(10)
    rand_ind = list(range(len(location_imgs)))
    random.shuffle(rand_ind)
    rand_ind.extend(rand_ind[0:12])  # 随机坐标

    # 按从0到11的顺序为每个参考设置查询切片
    rand_flags = []
    flag_dicts = {}  # 每个flag对应的随机flag
    for i, i_flag in tqdm(enumerate(location_imgs)):
        pano_img = location_imgs[i]
        pano_info = decom_pano_name(pano_img)
        lon, lat = pano_info['lon'], pano_info['lat']

        rand_flag = ''
        for k in range(slice_num):
            is_in_range = True
            s_k = i + k
            while is_in_range:
                select_flag = location_imgs[rand_ind[s_k]]
                ng_info = decom_pano_name(select_flag)
                ng_flag = ng_info['flag']
                tl_path = osp.join(sat_dir, ng_flag, 'tl_pos.txt')
                ref_range = np.loadtxt(tl_path, delimiter=' ')
                tl_lon, tl_lat = ref_range[0]
                br_lon, br_lat = ref_range[1]
                is_in_range = (lon < br_lon) & (lon > tl_lon) & (lat < tl_lat) & (lat > br_lat)
                if is_in_range:
                    print('find a fail instance')
                    s_k = random.randint(0, len(location_imgs) - 1)
                else:
                    is_in_range = False
                    rand_flag += (' ' + ng_flag)
        rand_flags.append(rand_flag)
        flag_dicts[pano_info['flag']] = rand_flag

    with open(nfa_valid_path, "w+") as file:
        for flag in rand_flags:
            file.write(flag + "\n")

    # 根据slice_cross.txt 生成 null_slice_single.txt
    slice_path = osp.join(pro_dir, city, 'slice_cross.txt')
    slice_imgs = []
    with open(slice_path, 'r') as f:
        grd_names = f.readlines()
    pano_datas = [data[:-1] for data in grd_names]
    slice_imgs.extend([data.split()[0] for data in pano_datas])

    rand_slice = []
    for slice in slice_imgs:
        flag, i_name = slice.split('/')
        i = int(osp.splitext(i_name)[0])
        rand_flag = flag_dicts[flag].split()[i]
        rand_slice.append(rand_flag)
    with open(rand_slice_path, "w+") as file:
        for flag in rand_slice:
            file.write(flag + "\n")


# 绘制main20设置得到背景测试下，Slice-Loc定位结果
def main21(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = True

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    save_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_H0'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_H0_plot'
    pro_dir = '/media/xmt/新加卷1/Data/SkyMap_Process_0'

    city = 'Chicago'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    txt_name = 'result.txt'
    save_txt = osp.join(save_dir, city, 'result.txt')  # flag, lxy, err, NFA,
    with open(save_txt, 'r') as f:
        lines = f.readlines()

    slice_loc_path = osp.join(save_dir, city, 'SliceLoc')
    loc_file = 'LocRes'
    loc_dir = osp.join(plot_dir, city, loc_file)
    if not osp.exists(loc_dir):
        os.makedirs(loc_dir)
    heatmap_dir = osp.join(plot_dir, city, 'Heatmap')
    if not osp.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    csat_dir = osp.join(plot_dir, city, 'ConSat')
    if not osp.exists(csat_dir):
        os.makedirs(csat_dir)

    slice_heat_dir = osp.join(save_dir, city, 'Heatmap')

    # same_area_balanced_train.txt
    data_txt = 'pano_label_balanced.txt'
    data_path = osp.join(pro_dir, city, data_txt)
    with open(data_path, 'r') as f:
        grd_datas = f.readlines()
    flag_dict = {}
    for d in grd_datas:
        grd_name = d.split()[0]
        info = decom_pano_name(grd_name)
        flag = info['flag']
        flag_dict[flag] = {
            'data': d[:-1]
        }

    # 绘制定位真值，slice定位结果，
    if plot_loc:
        for line1 in lines:
            # flag, lx, ly, err, nfa, dist1, dist2
            flag, slx, sly, err, nfa, _, _ = line1.split()
            slx, sly, err, nfa = float(slx), float(sly), float(err), float(nfa)

            _, _, x_rand, y_rand = flag_dict[flag]['data'].split()
            x_rand, y_rand = shift_range * float(x_rand), shift_range * float(y_rand)

            # err比err2 至少小5， NFA < -1
            if nfa > 0:
                continue

            # 加载原卫星图像
            raw_sat_path = osp.join(raw_dir, city, 'sat_img', flag, 'satellite.jpg')
            with PIL.Image.open(raw_sat_path, 'r') as SatMap:
                sat_map0 = SatMap.convert('RGB')
            sat_map0 = sat_map0.transform(
                sat_map0.size, PIL.Image.AFFINE,
                (1, 0, -x_rand,
                 0, 1, -y_rand),
                resample=PIL.Image.BILINEAR)
            sat_map0 = TF.center_crop(sat_map0, crop_size)  # 偏移后进行中心采样
            sat1 = np.array(sat_map0)
            sat0 = np.array(sat_map0)

            # 在exp_sat上绘制loc点与end点
            mat_path = osp.join(slice_loc_path, flag + '.npy')
            loc_mat = np.load(mat_path)
            # 将Slice定位的结果: x, y, ori绘制上去
            for pts in loc_mat:
                lx, ly, ex, ey, _, _, is_inlier = pts
                ld = np.sqrt((ex - lx) ** 2 + (ey - ly) ** 2)
                dx, dy = (ex - lx) / ld, (ey - ly) / ld
                ex_, ey_ = lx + 15 * dx, ly + 15 * dy
                if is_inlier:
                    sat0 = cv2.line(sat0, pt1=(int(lx), int(ly)),
                                    pt2=(int(ex_), int(ey_)),
                                    color=(0, 255, 0), thickness=3)
                else:
                    sat0 = cv2.line(sat0, pt1=(int(lx), int(ly)),
                                    pt2=(int(ex_), int(ey_)),
                                    color=(200, 0, 0), thickness=3)
                sat0 = cv2.circle(sat0, center=(int(lx), int(ly)), radius=3,
                                  thickness=1,
                                  color=(255, 0, 0))

            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=1,
                              thickness=-1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(sat1.shape[0] / 2 + x_rand), int(sat1.shape[1] / 2 + y_rand)), radius=3,
                              thickness=1,
                              color=(0, 0, 255))  # 真值
            sat1 = cv2.circle(sat1, center=(int(slx), int(sly)), radius=3,
                              thickness=-1,
                              color=(255, 0, 0))  # Slice-Loc

            img_path = osp.join(loc_dir, flag + '.png')
            s0_, s1_ = crop_img(sat0), crop_img(sat1)
            ccat_img = cv2.hconcat([s0_, s1_])
            ccat_img = cv2.cvtColor(ccat_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, ccat_img)

    # slice_txt_path = osp.join(pro_dir, city, 'null_slice_test.txt')
    # with open(slice_txt_path, 'r') as f:
    #     null_datas = f.readlines()
    # new_null_slice_txt = osp.join(pro_dir, city, 'H0_null_slice_test.txt')
    # new_data_txt = osp.join(pro_dir, city, 'H0_pano_label_balanced.txt')
    #
    # # 筛选出NFA<0的结果并重新保存 # H0_null_slice_test.txt, H0_pano_label_balanced.txt
    # res_flags = []
    # for i, line in enumerate(lines):
    #     flag, slx, sly, err, nfa, _, _ = line.split()
    #     nfa = float(nfa)
    #     if nfa < 0:
    #         res_flags.append(flag)
    #
    # with open(new_null_slice_txt, "w+") as file:
    #     for i, flag in enumerate(res_flags):
    #         file.write(null_datas[i])
    # with open(new_data_txt, "w+") as file1:
    #     for i, flag in enumerate(res_flags):
    #         file1.write(flag_dict[flag]['data'] + "\n")


# 根据背景测试的结果，估计背景分布的参数
# 设置背景分布为双直线分段分布
# 背景参数结果保存为
def main22(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = True

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    save_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_H0_0'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_H0_plot'
    pro_dir = '/media/xmt/新加卷1/Data/SkyMap_Process_0'

    city = 'Chicago'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    txt_name = 'result.txt'
    save_txt = osp.join(save_dir, city, 'result.txt')  # flag, lxy, err, NFA,
    with open(save_txt, 'r') as f:
        lines = f.readlines()

    # same_area_balanced_train.txt
    data_txt = 'pano_label_balanced.txt'
    data_path = osp.join(pro_dir, city, data_txt)
    with open(data_path, 'r') as f:
        grd_datas = f.readlines()
    flag_dict = {}
    for d in grd_datas:
        grd_name = d.split()[0]
        info = decom_pano_name(grd_name)
        flag = info['flag']
        flag_dict[flag] = {
            'data': d[:-1]
        }

    thetas = []
    for line in lines:
        _, _, _, _, theta = line.split()
        theta = float(theta)
        thetas.append(theta)
    thetas = np.array(thetas)
    mean = np.mean(thetas)
    std_dev = np.std(thetas, ddof=1)
    print('mean, std:', mean, std_dev)

    bin_num = 600
    thetas_abs = np.abs(thetas)
    hist, bin_edges = np.histogram(thetas_abs, bins=bin_num)
    y_data = hist
    x_data = bin_edges[1:]

    data = np.column_stack((x_data, y_data))

    # 使用 SVM 进行聚类
    svm = SVC(kernel='linear', C=1.0)
    # 这里假设你知道数据分为两类，可以给数据打上标签：A类（0），B类（1）
    theta_edge = 47
    num1 = np.sum(x_data < theta_edge)
    num2 = len(x_data) - num1
    svm.fit(data, np.concatenate((np.zeros(num1), np.ones(num2))))

    # 获取分类结果
    labels = svm.predict(data)

    # 将数据按分类标签分组
    data_a = data[labels == 0]
    data_b = data[labels == 1]

    # 使用线性回归分别拟合A类和B类数据的直线
    regressor_a = LinearRegression()
    regressor_b = LinearRegression()

    # 对A类数据拟合直线
    regressor_a.fit(data_a[:, 0].reshape(-1, 1), data_a[:, 1])
    # 对B类数据拟合直线
    regressor_b.fit(data_b[:, 0].reshape(-1, 1), data_b[:, 1])

    # 获取拟合的直线参数
    slope_a, intercept_a = regressor_a.coef_[0], regressor_a.intercept_
    slope_b, intercept_b = regressor_b.coef_[0], regressor_b.intercept_

    # 将直线拟合的结果保存
    save_txt = osp.join(pro_dir, city, 'line_param.txt')
    param_mat = np.array([[slope_a, intercept_a], [slope_b, intercept_b]])
    np.savetxt(save_txt, param_mat)

    # # 输出拟合结果
    # print(f"A类直线参数: 斜率 = {slope_a}, 截距 = {intercept_a}")
    # print(f"B类直线参数: 斜率 = {slope_b}, 截距 = {intercept_b}")
    #
    # # 可视化结果
    # plt.scatter(data_a[:, 0], data_a[:, 1], color='blue', label='A')
    # plt.scatter(data_b[:, 0], data_b[:, 1], color='red', label='B')
    #
    # # 绘制拟合的直线
    # x_vals = np.linspace(0, 10, 100)
    # y_vals_a = slope_a * x_vals + intercept_a
    # y_vals_b = slope_b * x_vals + intercept_b
    # plt.plot(x_vals, y_vals_a, 'b-', label=f'A: y = {slope_a:.2f}x + {intercept_a:.2f}')
    # plt.plot(x_vals, y_vals_b, 'r-', label=f'B: y = {slope_b:.2f}x + {intercept_b:.2f}')
    #
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.title('SVM聚类和直线拟合')
    # plt.show()

    pass


# 根据背景测试的结果，估计背景分布的参数
# 设置背景分布为正态分布 + 均匀分布
def main23(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = True

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    save_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_Exp'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Null_H0_plot'
    pro_dir = '/media/xmt/新加卷1/Data/SkyMap_Process_0'

    city = 'Sydney'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    txt_name = 'result.txt'
    save_txt = osp.join(save_dir, city, 'result.txt')  # flag, lxy, err, NFA,
    with open(save_txt, 'r') as f:
        lines = f.readlines()

    # same_area_balanced_train.txt
    data_txt = 'pano_label_balanced.txt'
    data_path = osp.join(pro_dir, city, data_txt)
    with open(data_path, 'r') as f:
        grd_datas = f.readlines()
    flag_dict = {}
    for d in grd_datas:
        grd_name = d.split()[0]
        info = decom_pano_name(grd_name)
        flag = info['flag']
        flag_dict[flag] = {
            'data': d[:-1]
        }

    thetas = []
    for line in lines:
        _, _, _, _, theta = line.split()
        theta = float(theta)
        thetas.append(theta)
    thetas = np.array(thetas)
    thetas = np.vstack((thetas, -thetas))

    r_the = 38
    l_the = 2 * np.mean(thetas) - r_the

    # 使用 scipy.stats.norm.fit 来拟合正态分布参数
    params = norm.fit(thetas)
    mean, std_dev = params
    save_txt = osp.join(pro_dir, 'norm_param.txt')
    param_mat = np.array([mean, std_dev, r_the, l_the])
    np.savetxt(save_txt, param_mat)


# 统计消融实验结果
def main24(argv):
    shift_range = 160
    crop_size = 640
    NFA_thr = -1

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    exp_res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Exp_res'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Plot_res'
    compare_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Compare_res'
    is_cross = False
    ori_list = ['45.0']  # ['0.0', '20.0', '45.0']
    if is_cross:
        dir_label = 'cross_'
        city_list = ['Sydney', 'Chicago', 'Johannesburg']
    else:
        dir_label = 'same_'
        city_list = ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']

    txt_name = 'result.txt'
    txt_ccvpe = 'result_ccvpe.txt'
    txt_hcnet = 'result_hcnet.txt'

    for ori in ori_list:
        res_dir = osp.join(exp_res_dir, dir_label + ori)
        if not osp.exists(res_dir):
            continue
        print('start processing ', dir_label + ori)
        mat_sliceloc = []
        mat_ccvpe = []

        # 读取文件中每个城市的结果并统计
        for city in city_list:
            data_path = osp.join(res_dir, city, txt_name)
            # 加载result.txt中的数据
            with open(data_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    flag, lx, ly, lerr, nfa, oerr, mloc, mori, _ = line.split()
                    # flag, lx, ly, err, nfa= line.split()
                    lx, ly, lerr, nfa, oerr = float(lx), float(ly), float(lerr), float(nfa), abs(float(oerr))
                    mloc, mori = float(mloc), abs(float(mori))
                    if lx > 0 and lx < 640 and ly > 0 and ly < 640:
                        mat_sliceloc.append([lx, ly, lerr, nfa, oerr, mloc, mori])

        mat_sliceloc = np.array(mat_sliceloc)
        nfa_valid = mat_sliceloc[:, 3] < NFA_thr
        # nfa_valid = np.ones(len(mat_sliceloc), dtype=bool)
        RoP = round(np.sum(nfa_valid) / len(nfa_valid), 4) * 100
        print('all num / in num / PoR:', len(nfa_valid), np.sum(nfa_valid), RoP)

        print('output slice-loc result:')
        v_lerrs = mat_sliceloc[nfa_valid, 2]
        v_oerrs = mat_sliceloc[nfa_valid, 4]
        print('the nfa location result:')
        lm3, lp3 = statistic_data(v_lerrs)
        print('the nfa orientation result:')
        om3, op3 = statistic_data(v_oerrs)
        print('the mean localization result:')
        m_lerrs = mat_sliceloc[:, 5]
        statistic_data(m_lerrs)
        print('the mean orientation result:')
        m_oerrs = mat_sliceloc[:, 6]
        statistic_data(m_oerrs)
        print('=======================================\n')


# 比较三种背景模型分布的粗差剔除效果
def main25(argv):
    shift_range = 160
    crop_size = 640
    plot_loc = False
    NFA_thr = 0

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    exp_res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Exp_res'
    plot_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc/Plot_temp'
    is_cross = True
    if is_cross:
        dir_label = 'cross_'
    else:
        dir_label = 'same_'

    txt_name = 'result.txt'
    # txt_name = 'result_uniform.txt'
    # txt_name = 'result_guassi.txt'
    # txt_name = 'result_line2.txt'

    ori = '0.0'
    dir_label += ori
    save_dir = osp.join(exp_res_dir, dir_label)
    plot_dir = osp.join(plot_dir, dir_label)
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_45'

    city = 'Chicago'  # ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    save_txt = osp.join(save_dir, city, txt_name)  # flag, lxy, err, NFA,

    slice_loc_path = osp.join(save_dir, city, 'SliceLoc')
    loc_file = 'LocRes'
    loc_dir = osp.join(plot_dir, city, loc_file)
    if not osp.exists(loc_dir):
        os.makedirs(loc_dir)

    # 加载result.txt中的数据
    nfas = []
    errs = []
    oerrs = []
    in_nums = []
    with open(save_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            flag, lx, ly, err, nfa, oerr, in_num = \
                temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[8]

            avg_err = float(temp[6])
            # flag, lx, ly, err, nfa= line.split()
            lx, ly, err, nfa, in_num = float(lx), float(ly), float(err), float(nfa), int(in_num)
            if lx < 0 or lx > 640 or ly < 0 or ly > 640:
                continue
            nfas.append(float(nfa))
            errs.append(float(err))
            oerrs.append(abs(float(oerr)))
            in_nums.append(in_num)
    errs = np.array(errs)
    nfas = np.array(nfas)
    oerrs = np.array(oerrs)
    in_nums = np.array(in_nums)
    save_path = osp.join(plot_dir, city, 'nfa-locerr.png')
    plot_err_nfa(errs, nfas, save_path)

    nfa_valid_ = nfas < NFA_thr
    # nfa_valid = np.ones(len(errs), dtype=bool)
    nfa_valid = nfa_valid_
    print('mean:', np.mean(errs[nfa_valid]))
    print('ratio: ', np.sum(nfa_valid_) / len(nfas))
    print('output slice-loc result:')
    print('all num / in num / PoR:', len(nfa_valid), np.sum(nfa_valid), round(np.sum(nfa_valid) / len(nfa_valid), 4))
    v_lerrs = errs[nfa_valid]
    print('the location result:')
    statistic_data(v_lerrs)
    v_oerrs = oerrs[nfa_valid]
    print('the orientation result:')
    statistic_data(v_oerrs)
    print('=======================================\n')

    # 绘制比例折线图
    nfa_dict = {}
    num = 21
    ratios = []
    for d in range(2, num, 2):
        s_ind = (d - 2 < errs) & (errs < d)
        errs_ = errs[s_ind]
        nafs_ = nfas[s_ind]
        r0 = np.sum(nafs_ > NFA_thr) / len(nafs_)
        rm1 = np.sum(nafs_ > NFA_thr - 1) / len(nafs_)
        nfa_dict[str(d)] = r0
        print(r0, ' ', rm1)
        ratios.append([r0, rm1])
    ratios = np.array(ratios)
    save_path = osp.join(plot_dir, city, 'nfa-ratio-line.png')
    plot_nfa_ratio(ratios, save_path)
    print('=======================================\n')

    # 比较小于10米的找回率，精度等指标
    prd_valids = nfa_valid
    gt_valids = errs < 10
    statistics_valid_res(prd_valids, gt_valids)


# 统计SkyMap中每个城市的全景影像覆盖的范围
def area_estimat():
    city = 'Chicago'
    slice_flag = 'slice'
    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    pano_dir = osp.join(raw_dir, city, 'panorama')
    sat_dir = osp.join(raw_dir, city, 'sat_img')

    files = [f for f in os.listdir(pano_dir) if f.endswith('.jpg')]

    for file in files:
        file_path = os.path.join(pano_dir, file)
        img_info = decom_pano_name(file_path)
        # 创建坐标转换的类
        sat_path = osp.join(sat_dir, img_info['flag'], 'satellite.jpg')
        sat_size = cv2.imread(sat_path).shape[0]
        tl_path = osp.join(sat_dir, img_info['flag'], 'tl_pos.txt')
        i_trans = GeoTrans(json_path=None, tl_path=tl_path, sat_size=sat_size)


def temp_range_vigor():
    vigor_path = '/media/xmt/563A16213A15FEA5/XMT/Datas/VIGOR/NewYork/panorama'
    name_list = [f for f in os.listdir(vigor_path) if f.endswith('.jpg')]

    lats = []
    lons = []
    for name in name_list:
        info = decom_pano_name_vigor(name)
        lats.append(info['lat'])
        lons.append(info['lon'])
    lons = np.array(lons)
    lats = np.array(lats)
    min_lon, max_lon = np.min(lons), np.max(lons)
    min_lat, max_lat = np.min(lats), np.max(lats)
    lat_list = np.linspace(min_lat, max_lat, 4)
    d1 = {
        'lat': max_lat,
        'lng': max_lon
    }
    d2 = {
        'lat': max_lat,
        'lng': min_lon
    }
    d3 = {
        'lat': min_lat,
        'lng': min_lon
    }
    d4 = {
        'lat': min_lat,
        'lng': max_lon
    }
    data = [d1, d2, d3, d4]
    json_path = f"/media/xmt/563A16213A15FEA5/XMT/Datas/range_data/range_yk.json"
    with open(json_path, 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # for i in range(1, 4):
    #     max_lat = lat_list[i]
    #     d1 = {
    #         'lat':max_lat,
    #         'lng':max_lon
    #     }
    #     d2 = {
    #         'lat': max_lat,
    #         'lng': min_lon
    #     }
    #     d3 = {
    #         'lat': min_lat,
    #         'lng': min_lon
    #     }
    #     d4 = {za
    #         'lat': min_lat,
    #         'lng': max_lon
    #     }
    #     data = [d1, d2, d3, d4]
    #     json_path = f"/media/xmt/563A16213A15FEA5/XMT/Datas/range_data/range_{i}.json"
    #     with open(json_path, 'w+', encoding='utf-8') as f:
    #         json.dump(data, f, ensure_ascii=False, indent=2)


# 检查下好的数据并输出需要重下的uid文本文件
def check_flag():
    f_dir = ''
    downloads = [f for f in os.listdir(f_dir) if f.endswith('.json')]

    city = 'Chicago'
    p_path = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    pano_dir = osp.join(raw_dir, city, 'panorama')

    p_flags_ = [f for f in os.listdir(pano_dir) if f.endswith('.jpg')]
    p_flags = []  # VIGOR中的flag
    for name in p_flags_:
        file_path = os.path.join(pano_dir, name)
        img_info = decom_pano_name(file_path)
        p_flags.append(img_info['flag'])

    re_downloads = []
    for name in downloads:
        file_path = os.path.join(pano_dir, name)
        img_info = decom_pano_name(file_path)
        uid = img_info['flag']
        if not uid in p_flags:
            re_downloads.append(uid)

    txt_path = '/media/xmt/563A16213A15FEA5/XMT/Datas/range_data/re_download.txt'
    with open(txt_path, "w+") as file:
        for data in re_downloads:
            file.write(data + "\n")


# 统计Slice-Loc Weakly的结果
def main26(argv):
    shift_range = 160
    crop_size = 640
    NFA_thr = 0

    raw_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap'
    exp_res_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/Slice_Loc_Weakly'
    is_cross = True

    # dir_label = 'slice_cross_ransac'
    dir_label = 'slice_cross_pseudo'
    # dir_label = 'slice_cross'
    # dir_label = 'slice_cross_eccv'

    city_list = ['Sydney', 'Johannesburg', 'Chicago']

    txt_name = 'result.txt'

    res_dir = osp.join(exp_res_dir, dir_label)
    print('start processing ', dir_label)
    mat_sliceloc = []
    mat_ccvpe = []

    # 读取文件中每个城市的结果并统计
    for city in city_list:
        data_path = osp.join(res_dir, city, txt_name)
        # 加载result.txt中的数据
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                temp = line.split()
                flag, lx, ly, lerr, nfa, oerr = \
                    temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]
                avg_err = float(temp[6])
                # flag, lx, ly, err, nfa= line.split()
                lx, ly, lerr, nfa, oerr = float(lx), float(ly), float(lerr), float(nfa), abs(float(oerr))
                if lx > 0 and lx < 640 and ly > 0 and ly < 640:
                    mat_sliceloc.append([lx, ly, lerr, nfa, oerr, avg_err])

    mat_sliceloc = np.array(mat_sliceloc)
    nfa_valid = mat_sliceloc[:, 3] < NFA_thr
    # nfa_valid = np.ones(len(mat_sliceloc), dtype=bool)
    RoP = round(np.sum(nfa_valid) / len(nfa_valid), 4) * 100
    print('all num / in num / PoR:', len(nfa_valid), np.sum(nfa_valid), RoP)

    print('output slice-loc result:')
    v_lerrs = mat_sliceloc[nfa_valid, 2]
    v_oerrs = mat_sliceloc[nfa_valid, 4]
    print('the location result:')
    lm3, lp3 = statistic_data(v_lerrs)
    print('the orientation result:')
    om3, op3 = statistic_data(v_oerrs)
    print('=======================================\n')

    avg_errs = mat_sliceloc[nfa_valid, 5]
    print('the average location result:')
    statistic_data(avg_errs)

# 加载dress-d中结果，计算默认距离
# 用于训练VIGOR数据集
def gen_default_distance():
    city_list = ['Sydney', 'Chicago', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    # city_list = ['London']
    pro_dir = '/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap_Processing_45'
    dist_list = []

    for city in city_list:
        slice_path = osp.join(pro_dir, city, 'slice')
        flags = os.listdir(slice_path)
        for flag in flags:
            loc_path = osp.join(slice_path, flag, 'coordinate.npy')
            loc_data = np.load(loc_path)
            loc_data -= 640
            dist = np.linalg.norm(loc_data, axis=1)
            dist_list.append(np.mean(dist))

    dist_list = np.array(dist_list)
    print('mean dist: ', np.mean(dist_list))


if __name__ == "__main__":
    gen_default_distance()
    # temp_range_vigor() # 获取vigor的定位范围
    # analyse_vigor()
    # main6(sys.argv) # 多线程进行全景图像切分
    # main7(sys.argv) # 切片模型的文本文件生成

    # main9(sys.argv) # 绘制NFA散点分布与折线图

    # main18(sys.argv)  # 统计三种方法的对比结果
    # main23(sys.argv)
    # main20(sys.argv)
    # main25(sys.argv) # 对比三种背景分布的小于10米的召回率等的指标

    # main26(sys.argv) # 统计弱监督的方法的效果
