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
import argparse

import threading
from multiprocessing import Pool
import PIL.Image
import PIL.ImageDraw
import torchvision.transforms.functional as TF

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
            coor_list.append(mv)
            output_file_path = os.path.join(p_path, str(i) + '.jpg')

            # 保存提取出的图像
            cv2.imwrite(output_file_path, out_img)
        coor_mat = np.array(coor_list)
        np.save(coor_path, coor_mat)

# 对每个城市生成相关的数据
def generate_city_data(city, raw_dir, output_dir, ori_range=20, crop_img=True, slice_num=12):
    print('processing city:', city)
    slice_flag = 'slice'
    pano_dir = osp.join(raw_dir, city, 'panorama')

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


# 根据txt中的内容生成打乱的Slice数据，用于训练Slice SkyMap
def generate_train_txt(flag_train_txt, train_txt, pro_path, slice_num=12):
    os.makedirs(osp.dirname(train_txt), exist_ok=True)
    shutil.copy(flag_train_txt, osp.dirname(train_txt))
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


def crop_img(image):
    cx, cy = image.shape[0] / 2, image.shape[0] / 2
    lenx, leny = 200, 200
    ux, uy, dx, dy = int(cx + lenx), int(cy + leny), int(cx - lenx), int(cy - leny)
    new_img = image[dy:uy, dx:ux, :]
    return new_img


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


def generate_slice_data(argv):
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='generate the slice-level data')

    parser.add_argument('--data_root', type=str, default=r'/media/xmt/563A16213A15FEA5/XMT/Datas/SkyMap',
                        help='path to the root folder of all dataset')
    parser.add_argument('--pro_dir', type=str, default=r'/media/xmt/563A16213A15FEA5/XMT/Datas/test_gen_slice_data',
                        help='path to the root folder of all processed sliced image')
    parser.add_argument('--slice_num', default=12, type=int, help='the number of sliced image')
    parser.add_argument('--ori_range', default=45, type=int, help='the range of orientation noise')

    args = vars(parser.parse_args())
    data_root = args['data_root']
    pro_dir = args['pro_dir']
    slice_num = args['slice_num']
    ori_range = args['ori_range']

    # citys = ['Chicago', 'Sydney', 'Johannesburg', 'Tokyo', 'Rio', 'London']
    citys = ['Chicago']

    random.seed(100)
    slice_flag = 'slice'
    split_flag = 'split_slice_loc'


    print('start slice panorama with ori range: ', ori_range)
    print('save the sliced data at: ', pro_dir)
    for city in citys:
        print('slicing the data of ', city)
        pano_names, rand_mat = generate_city_data(city, data_root, pro_dir, ori_range, crop_img=True,
                                                  slice_num=slice_num)

        # generate the slice-leve txt
        cross_txt = osp.join(data_root, split_flag, city, 'pano_label_balanced.txt')
        same_test_txt = osp.join(data_root, split_flag, city, 'same_area_balanced_test.txt')
        same_train_txt = osp.join(data_root, split_flag, city, 'same_area_balanced_train.txt')

        slice_cross_txt = osp.join(pro_dir, city, 'slice_cross.txt')
        slice_same_test_txt = osp.join(pro_dir, city, 'slice_same_test.txt')
        slice_same_train_txt = osp.join(pro_dir, city, 'slice_same_train.txt')

        pro_path = osp.join(pro_dir, city, slice_flag)

        generate_train_txt(cross_txt, slice_cross_txt, pro_path, slice_num)
        generate_train_txt(same_test_txt, slice_same_test_txt, pro_path, slice_num)
        generate_train_txt(same_train_txt, slice_same_train_txt, pro_path, slice_num)


if __name__ == "__main__":
    generate_slice_data(sys.argv)
