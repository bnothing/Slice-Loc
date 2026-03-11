import sys
import numpy as np
import math
from math import log10 as log10
from scipy.optimize import minimize
from shapely.geometry import Polygon
import cv2
from scipy.stats import norm

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


# 计算定位的点，与方向的点，返回 N*4 大小的矩阵
def gen_slice_loc_res(locs, cpt, gt_pt, ori):
    slice_num = locs.shape[0]
    # 根据ori旋转locs
    if ori != 0:
        for i in range(locs.shape[0]):
            locs[i] = rotate_point(locs[i,0], locs[i,1], cpt[0], cpt[1], -ori)

    for i in range(locs.shape[0]):
        locs[i] = rotate_point(locs[i,0], locs[i,1], gt_pt[0], gt_pt[1], ori)

    # 根据slice 方向生成尾点
    d = 10
    end_pts = []
    for s in range(slice_num):
        theta = s*np.pi/6
        e_pt = locs[s]+np.array([d*np.sin(theta), -d*np.cos(theta)])
        end_pts.append(e_pt)
    res = np.concatenate((locs, np.array(end_pts)), axis=1)
    return res


# 计算点到射线的距离
def point_to_ray_distance(P, A, d):
    # 点 P 的坐标
    x0, y0 = P
    # 射线的起点 A 的坐标
    x1, y1 = A
    # 射线的方向向量 d
    dx, dy = d

    # 计算向量 AP
    AP = np.array([x0 - x1, y0 - y1])
    # 射线方向向量 d
    D = np.array([dx, dy])

    # 射线方向向量的长度平方
    d_length_squared = np.dot(D, D)

    # 计算投影标量
    projection_scalar = np.dot(AP, D) / d_length_squared

    if projection_scalar < 0:
        # 投影点在射线方向的起点之前
        distance = np.linalg.norm(AP)
    else:
        # 投影点的坐标
        projection_point = np.array([x1, y1]) + projection_scalar * D
        # 计算点 P 到投影点的距离
        distance = np.linalg.norm(P - projection_point)

    return distance

# 计算点到直线的距离
def point_to_line_distance(P, A, d):
    x, y = P
    x1, y1 = A
    dx, dy = d
    # 点到直线的距离公式
    distance = abs((x - x1) * dy - (y - y1) * dx)
    return distance

# 根据各切片的视觉定位，进行可靠的相机定位
def location_camera_center(locs):

    # 生成各方向的方向信息，默认已消除方向误差
    slice_num = locs.shape[0]
    s_angle = 360/slice_num
    directions = []
    for s in range(slice_num):
        angle = np.radians(180 - s*s_angle)
        dx = np.sin(angle)
        dy = np.cos(angle)
        directions.append([dx, dy])
    int_pts = []
    for i in range(slice_num-1):
        for j in range(i+1, slice_num):
            if j-i == slice_num/2:
                continue
            # 计算交点
            x1, y1 = locs[i]
            x2, y2 = locs[j]
            dx1, dy1 = directions[i]
            dx2, dy2 = directions[j]

            A = np.array([[dx1, -dx2], [dy1, -dy2]])
            b = np.array([x2 - x1, y2 - y1])
            t, s = np.linalg.solve(A, b)
            intersection = (x1 + t * dx1, y1 + t * dy1)
            int_pts.append(intersection)
    # 两两遍历切片，交出相机位置
    int_pts = np.array(int_pts)
    err = int_pts - np.mean(int_pts,axis=0)
    dist = np.sqrt(np.sum(err**2, axis=1))
    val_ind = dist < 3*np.mean(dist)
    val_pts = int_pts[val_ind]
    return np.mean(val_pts,axis=0)

# 根据切片的定位结果进行相机位置定位
def location_camera_center2(locs):
    # 生成各方向的方向信息，默认已消除方向误差
    slice_num = locs.shape[0]
    s_angle = 360/slice_num
    directions = []
    for s in range(slice_num):
        angle = np.radians(180 - s*s_angle)
        dx = np.sin(angle)
        dy = np.cos(angle)
        directions.append([dx, dy])

    inter_mat = np.zeros((slice_num, slice_num, 2))
    thr = 5
    max_num = -1
    loc_pt = []
    int_pts = []
    in_set = []
    for i in range(slice_num-1):
        for j in range(i+1, slice_num):
            if j-i == slice_num/2:
                continue
            # 计算交点
            x1, y1 = locs[i]
            x2, y2 = locs[j]
            dx1, dy1 = directions[i]
            dx2, dy2 = directions[j]

            A = np.array([[dx1, -dx2], [dy1, -dy2]])
            b = np.array([x2 - x1, y2 - y1])
            t, s = np.linalg.solve(A, b)
            pt0 = (x1 + t * dx1, y1 + t * dy1)
            int_pts.append(pt0)
            inter_mat[i, j] = np.array(pt0)

            # 计算该交点到各切片射线之间的距离
            dists = []
            all_ind = np.arange(slice_num)
            for k in range(slice_num):
                A = locs[k]
                d = directions[k]
                dist = point_to_ray_distance(pt0, A, d) # 计算点到射线的距离
                # dist = point_to_line_distance(pt0, A, d) # 计算点到直线的距离
                dists.append(dist)
            dists = np.array(dists)
            in_ind = all_ind[dists < thr]
            in_num = np.sum(dists < thr)
            if in_num > max_num:
                loc_pt = pt0
                max_num = in_num
                in_set = in_ind
            pass

    inlier_pt = []
    for i in range(len(in_set)-1):
        for j in range(i+1, len(in_set)):
            if j-i == slice_num/2:
                continue
            inlier_pt.append(inter_mat[i, j])
    inlier_pt = np.array(inlier_pt)
    fit_pt = np.mean(inlier_pt, axis=0)
    return np.array(loc_pt), max_num
    # return fit_pt, max_num


class ErrorIndex:
    def __init__(self, e=0, i=0):
        self.eps = 1e-8
        self.err = e
        self.ind = i

    def __lt__(self, other):  # 真·小于
        if abs(self.err - other.err) > self.eps:
            return self.err < other.err
        return self.err < other.err

# 生成各切片方向的原本的夹角：与正北方向顺时针角度
def compute_directions(locs,prd_directs=None):
    slice_num = locs.shape[0]
    s_angle = 360 / slice_num
    directions = []
    if not prd_directs is None:
        ori = np.mean(prd_directs)
    else:
        ori = 90
    for s in range(slice_num):
        if prd_directs is None:
            angle = np.radians(s * s_angle)
        else:
            angle = np.radians(s * s_angle+ori-90)
        dx = np.sin(angle)
        dy = -np.cos(angle)
        directions.append([dx, dy])
    return np.array(directions)

# 计算向量与正北方向夹角
def calculate_angle_with_y_axis(vx, vy):
    # 处理v_x为零的情况
    if vx == 0:
        if vy > 0:
            return math.pi / 2
        elif vy < 0:
            return -math.pi / 2
        else:
            return 0  # 向量为零向量
    # 计算向量与x轴正方向的夹角
    theta = math.atan2(vy, vx)
    # 计算向量与y轴正方向的夹角
    phi = theta - math.pi / 2
    # 确保角度在[0, 2*pi)范围内
    if phi < 0:
        phi += 2 * math.pi
    return math.degrees(phi)

def angle_with_y_axis_negative(vx, vy):
    theta = math.atan2(vy, vx) + (math.pi if vy < 0 else 0)
    return math.degrees(theta)



def ray_intersection(ray1, ray2):
    """
    判断两射线是否相交，并返回交点（如果存在）。
    :param ray1: 第一条射线，格式为 (x1, y1, dx1, dy1)
    :param ray2: 第二条射线，格式为 (x2, y2, dx2, dy2)
    :return: (是否相交, 交点坐标或 None)
    """
    x1, y1, dx1, dy1 = ray1
    x2, y2, dx2, dy2 = ray2

    # 构建矩阵方程 A * [t1, t2]^T = B
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([x2 - x1, y2 - y1])

    try:
        # 求解方程组
        t = np.linalg.solve(A, B)
        t1, t2 = t

        # 判断 t1 和 t2 是否满足射线条件
        if t1 >= 0 and t2 >= 0:
            # 计算交点
            intersection_point = (x1 + t1 * dx1, y1 + t1 * dy1)
            return True, intersection_point
        else:
            return False, None
    except np.linalg.LinAlgError:
        # 矩阵不可逆（射线平行或重合）
        return False, None

# 使用直线进行相机位置采样
def generate_models(locs, directions):
    int_pts = []
    slice_ind = []
    slice_num = len(locs)
    for i in range(slice_num - 1):
        for j in range(i + 1, slice_num):
            if j - i == slice_num / 2:
                # 当两个射线平行时
                pt0 = (locs[i] + locs[j]) / 2
            else:
                # 计算交点
                x1, y1 = locs[i]
                x2, y2 = locs[j]
                dx1, dy1 = directions[i]
                dx2, dy2 = directions[j]

                A = np.array([[dx1, -dx2], [dy1, -dy2]])
                b = np.array([x2 - x1, y2 - y1])
                t, s = np.linalg.solve(A, b)
                pt0 = (x1 + t * dx1, y1 + t * dy1)
            int_pts.append(pt0)
            slice_ind.append([i, j])
    return np.array(int_pts), np.array(slice_ind)


# 使用射线相交进行相机位置采样
def generate_models2(locs, directions):
    int_pts = []
    slice_ind = []
    slice_num = len(locs)
    for i in range(slice_num - 1):
        for j in range(i + 1, slice_num):
            if j - i == slice_num / 2:
                # 两射线平行时
                vec1 = directions[i]
                vec2 = locs[j] - locs[i]
                theta0 = vector_angle(vec1, vec2)
                if abs(theta0) < np.pi/4: # 相对极角小于45°
                    pt0 = (locs[i] + locs[j]) / 2
                    int_pts.append(pt0)
                else:
                    pt0 = np.array([-1, -1])
                    int_pts.append(pt0)
            else:
                # 计算交点
                x1, y1 = locs[i]
                x2, y2 = locs[j]
                dx1, dy1 = directions[i]
                dx2, dy2 = directions[j]
                valid, pt0 = ray_intersection((x1, y1, dx1, dy1), (x2, y2, dx2, dy2))
                if valid:
                    int_pts.append(pt0)
                else:
                    pt0 = np.array([-1, -1])
                    int_pts.append(pt0)
            slice_ind.append([i, j])
    return np.array(int_pts), np.array(slice_ind)

# 生成所有定位模型: C(12,2)-6=60
# 交点、切片下标
# 定位时相机旋转角使用采样的射线旋转角度的均值
def generate_models3(locs, prd_oris):
    int_pts = []
    slice_ind = []
    cam_oris = []
    slice_num = len(locs)
    s_angle = 360 / slice_num
    for i in range(slice_num - 1):
        for j in range(i + 1, slice_num):
            if j - i == slice_num / 2:
                # 当两个射线平行时
                pt0 = (locs[i] + locs[j]) / 2
            else:
                # 计算交点
                x1, y1 = locs[i]
                x2, y2 = locs[j]
                hi = np.radians(i * s_angle + prd_oris[i] - 90)
                hj = np.radians(j * s_angle + prd_oris[j] - 90)
                dx1, dy1 = np.sin(hi), -np.cos(hi)
                dx2, dy2 = np.sin(hj), -np.cos(hj)

                A = np.array([[dx1, -dx2], [dy1, -dy2]])
                b = np.array([x2 - x1, y2 - y1])
                t, s = np.linalg.solve(A, b)
                pt0 = (x1 + t * dx1, y1 + t * dy1)
            int_pts.append(pt0)
            slice_ind.append([i, j])
            ori = (prd_oris[i]+prd_oris[j])/2 - 90
            cam_oris.append(ori)
    return np.array(int_pts), np.array(slice_ind), np.array(cam_oris)



# 计算点到各切片射线的距离
def pt2slice_rays(pt, locs, directions):
    dists = []
    for k in range(locs.shape[0]):
        A = locs[k]
        d = directions[k]
        dist = point_to_ray_distance(pt, A, d) # 计算点到射线的距离
        # dist = point_to_line_distance(pt, A, d)  # 计算点到直线的距离

        dists.append(dist)
    return np.array(dists)

# 计算两向量夹角的值
def vector_angle(v1, v2):
    # 计算点积
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    # 计算向量的模长
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    # 防止除以零的错误（虽然理论上模长不应该为零，但数值计算中可能会有误差）
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 180
    # 计算夹角的余弦值
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    if cosine_angle > 1.:
        cosine_angle = 1.
    if cosine_angle < -1.:
        cosine_angle = -1.
    # 计算并返回夹角（使用arccos将余弦值转换为角度）
    angle = math.acos(cosine_angle)
    return angle

# 计算各切片的角度误差
def rotation_error(pt, locs, oris):
    slice_num = locs.shape[0]
    s_angle = 360 / slice_num
    dists = []
    if oris is None:
        ori = 0.0
    else:
        ori = np.mean(oris)-90
    for k in range(locs.shape[0]):
        heading = np.radians(k * s_angle + ori)
        view_vec = np.array([-np.sin(heading), np.cos(heading)])
        # north_vec = np.array([0, -1])
        # theta1 = vector_angle(north_vec, view_vec)
        ray_vec = locs[k]-pt
        # theta2 = vector_angle(north_vec, ray_vec)
        theta0 = vector_angle(view_vec, ray_vec)
        dists.append(np.degrees(theta0))
    return np.abs(np.array(dists))

# 使用相机旋转计算误差，旋转来自各自的切片
def rotation_error2(pt, locs, cam_ori, prd_oris):
    slice_num = locs.shape[0]
    s_angle = 360 / slice_num
    dists = []
    tt = []
    for k in range(locs.shape[0]):
        heading = np.radians(k * s_angle + prd_oris[k]-90)
        view_vec = np.array([-np.sin(heading), np.cos(heading)])
        ray_vec = locs[k]-pt
        theta0 = vector_angle(ray_vec, view_vec)
        ori_err = np.degrees(theta0)
        dists.append(ori_err * (prd_oris[k]-cam_ori-90))
        tt.append([ori_err, prd_oris[k]-cam_ori-90])
    return np.abs(np.array(dists)), np.abs(np.array(tt))

# 两种角度误差，旋转来自切片旋转的均值
def rotation_error3(pt, locs, prd_oris):
    slice_num = locs.shape[0]
    s_angle = 360 / slice_num
    dists = []
    tt = []
    if prd_oris is None:
        ori = 0.0
    else:
        ori = np.mean(prd_oris)-90
    for k in range(locs.shape[0]):
        heading = np.radians(k * s_angle + ori)
        view_vec = np.array([-np.sin(heading), np.cos(heading)])
        ray_vec = locs[k]-pt
        theta0 = vector_angle(ray_vec, view_vec)
        ori_err = np.degrees(theta0)
        dists.append(max(ori_err, (prd_oris[k]-ori-90)))
        tt.append([ori_err, prd_oris[k]-ori-90])
    return np.abs(np.array(dists)), np.abs(np.array(tt))


# 计算NFA, 返回最小NFA所在的，其中的
def FindBest(mchErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample):
    sInd = 2
    bestInd = ErrorIndex(sys.maxsize, sInd)
    n = len(mchErr)
    for k in range(sInd+1, n+1): # 此处的k为第k个元素
        if mchErr[k-1].err > thr:
            break
        logAlpha = log10(np.pi)+2*log10(mchErr[k-1].err)-logSrch
        logNFA = loge0+logAlpha*(k-sInd)+aryLogCNK[k]+aryLogCKSample[k]
        temp = ErrorIndex(logNFA, k)
        if temp.err < bestInd.err:
            bestInd.err = temp.err
            bestInd.ind = temp.ind
    return bestInd

# 使用自定义的分布
def FindBest_theta(mchErr, thr, loge0, aryLogCNK, aryLogCKSample, h0):
    sInd = 2
    bestInd = ErrorIndex(sys.maxsize, sInd)
    n = len(mchErr)
    for k in range(sInd+1, n+1): # 此处的k为第k个元素
        if mchErr[k-1].err > thr:
            break
        alpha = h0.probability2(mchErr[k-1].err)
        logAlpha = log10(alpha)
        logNFA = loge0+logAlpha*(k-sInd)+aryLogCNK[k]+aryLogCKSample[k]
        temp = ErrorIndex(logNFA, k)
        if temp.err < bestInd.err:
            bestInd.err = temp.err
            bestInd.ind = temp.ind
    return bestInd

# 使用的正态分布
def FindBest_theta_normal(mchErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample):
    sInd = 2
    bestInd = ErrorIndex(sys.maxsize, sInd)
    n = len(mchErr)
    mean = 0
    std = 59.4683
    y = norm.cdf(0, mean, std)
    if mean < 0:
        alpha0 = 2 * y - 1
    else:
        alpha0 = 1 - 2 * y
    for k in range(sInd+1, n+1): # 此处的k为第k个元素
        if mchErr[k-1].err > thr:
            break
        x_ = mchErr[k-1].err
        y = norm.cdf(x_, mean, std)
        if x_ > mean:
            alpha = 2 * y - 1
        else:
            alpha = 1 - 2 * y
        logAlpha = log10(alpha)
        logNFA = loge0+logAlpha*(k-sInd)+aryLogCNK[k]+aryLogCKSample[k]
        temp = ErrorIndex(logNFA, k)
        if temp.err < bestInd.err:
            bestInd.err = temp.err
            bestInd.ind = temp.ind
    return bestInd

# 使用随机分布
def FindBest_theta_uniform(mchErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample):
    sInd = 2
    bestInd = ErrorIndex(sys.maxsize, sInd)
    n = len(mchErr)
    for k in range(sInd+1, n+1): # 此处的k为第k个元素
        if mchErr[k-1].err > thr:
            break
        logAlpha = log10(mchErr[k-1].err)-logSrch
        logNFA = loge0+logAlpha*(k-sInd)+aryLogCNK[k]+aryLogCKSample[k]
        temp = ErrorIndex(logNFA, k)
        if temp.err < bestInd.err:
            bestInd.err = temp.err
            bestInd.ind = temp.ind
    return bestInd


# 计算组合数C(k, n) 取10的对数
def logCombi(k, n):
    if k>=n or k<=0:
        return 0.
    kt = k
    if n-k<k:
        kt = n-k
    r = 0.
    for i in range(1,kt+1):
        r += math.log10(n-i+1) - math.log10(i)
    return r

# 计算组合数组： logCombi(:, n)
def CalLogCNK(n):
    res = np.zeros((n+1, 1))
    for k in range(n+1):
        res[k] = logCombi(k, n)
    return res

# 计算组合数组： logCombi(k, :)
def CalLogCKSample(k, nmax):
    res = np.zeros(nmax+1)
    for i in range(nmax+1):
        res[i] = logCombi(k, i)
    return res

def eliminate_zeros(errors, start_index):
    if errors[start_index].err > 0.:
        return
    for i in range(start_index, len(errors)):
        if errors[i].err <= 0.:
            errors[i].err += i*1e-4

def estimate_search_area2(locs):
    max_x, max_y = np.max(locs, axis=0)
    min_x, min_y = np.min(locs, axis=0)
    min_len = min((max_x-min_x), (max_y-min_y))
    return min_len**2

def estimate_search_area(locs):
    # 使用Polygon创建多边形
    poly = Polygon(locs)
    # 计算并输出面积
    area = poly.area
    return area

# 根据射线与内点下标，最优化求出距离最小的点
def find_optimal_point(pts, directions):
    # 定义直线的参数
    # 这里假设每条直线由两个点确定
    end_pts = pts + directions
    lines = np.concatenate((pts, end_pts), axis=1)

    def distance_to_line(point, line):
        """计算点到直线的距离"""
        x1, y1, x2, y2 = line
        x0, y0 = point
        # 直线的系数
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return abs(A * x0 + B * y0 + C) / np.sqrt(A ** 2 + B ** 2)

    def total_distance(point):
        """计算给定点到所有直线的距离之和"""
        return sum(distance_to_line(point, line) for line in lines)

    init_pt = np.mean(pts, axis=0)
    opt_res = minimize(total_distance, init_pt)
    # if opt_res.success:
    return opt_res.x
    # else:
    #     return None

def find_opt_pt(pt0, locs, directoins):
    def point_to_lines_distance_sum(point, lpts, ldirs):
        x, y = point
        total_distance = 0
        for line_point, direction in zip(lpts, ldirs):
            x1, y1 = line_point
            dx, dy = direction
            # 计算点到直线的距离
            distance = abs((x - x1) * dy - (y - y1) * dx)
            total_distance += distance
        return total_distance

    result = minimize(
        fun=point_to_lines_distance_sum,
        x0=np.array(pt0),
        args=(locs,directoins,),
        method='L-BFGS-B'  # 使用约束优化方法
    )
    return result.x, result.fun

# 根据切片的定位结果进行NFA计算，
def location_camera_center_NFA(locs, src_area=None, nfa_thr=-1, out_nfa=False, reset_err=False,prd_directs=None,ray_fit=True):
    # 遍历采样，拟合偏移模型
    dataSize = locs.shape[0]
    nSample = 2
    aryLogCNK = CalLogCNK(dataSize)
    aryLogCKSample = CalLogCKSample(nSample, dataSize)
    locErr = [None] * dataSize

    # orsa 初始化
    loge0 = log10(dataSize-nSample)
    if src_area is None:
        src_area = estimate_search_area(locs)

    logSrch = log10(src_area)
    thr = sys.maxsize
    minNFA = ErrorIndex(np.array([sys.maxsize]), 100)
    res_pt = np.array([-1, -1])

    directions = compute_directions(locs,prd_directs)
    if ray_fit:
        loc_pts, s_inds = generate_models(locs, directions) # 直线相交计算模型
    else:
        loc_pts, s_inds = generate_models2(locs, directions) # 射线相交计算模型
    max_dist = 0
    inlierIdx = np.full(dataSize, False)
    sloc_errs = np.zeros(dataSize)
    for i, pt in enumerate(loc_pts):
        if pt[0] == -1:
            continue
        errs = pt2slice_rays(pt, locs, directions) # 计算距离
        for ind, data in enumerate(errs):
            locErr[ind] = ErrorIndex(data, ind)
        if reset_err:
            locErr[s_inds[i,0]].err=0
            locErr[s_inds[i, 1]].err = 0
        locErr = sorted(locErr)
        eliminate_zeros(locErr, nSample)
        best = FindBest(locErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample)
        if best.err < minNFA.err:
            minNFA.err = best.err
            minNFA.ind = best.ind
            max_dist = errs[best.ind-1]
            inlierIdx = np.full(dataSize, False)
            sloc_errs = np.zeros(dataSize)
            res_pt = pt
            for j in range(best.ind):
                inlierIdx[locErr[j].ind] = True
            for kk in range(len(locErr)):
                sloc_errs[locErr[kk].ind] = locErr[kk].err
        # 绘制定位示意图
        # 绘制定位点，绘制12根射线
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        image = cv2.circle(image, center=(int(pt[0]), int(pt[1])), radius=3,
                          thickness=-1,
                          color=(255, 0, 0))
        # B G R
        for tt, loc in enumerate(locs):
            lx, ly = loc
            dx, dy = directions[tt]
            ex_, ey_ = lx + 15 * dx, ly + 15 * dy
            if inlierIdx[tt] == True:
                # 内点为绿色
                image = cv2.line(image, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 255, 0), thickness=3)
                # 否则为红色
            else:
                image = cv2.line(image, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 0, 255), thickness=3)
            image = cv2.circle(image, center=(int(loc[0]), int(loc[1])), radius=3,
                              thickness=1,
                              color=(255, 0, 0))
        pass
    # 模型都不可用
    if minNFA.err == sys.maxsize:
        res_pt = np.mean(locs, axis=0)
        sloc_errs = pt2slice_rays(res_pt, locs, directions) # 计算距离
    in_num = np.sum(inlierIdx)
    in_locs = locs[inlierIdx]
    in_dires = directions[inlierIdx]
    if in_num > 2:
        res_pt, info = find_opt_pt(res_pt, in_locs, in_dires)
    if out_nfa:
        return res_pt, inlierIdx, minNFA.err < nfa_thr, minNFA.err, sloc_errs
    else:
        return res_pt, inlierIdx, minNFA.err < nfa_thr

# 使用角度theta进行判断
def estimate_camera_pose(locs,
                                out_nfa=False,
                                reset_err=True,
                                prd_directs=None,
                                ray_fit=True,
                                h0=None):
    # 遍历采样，拟合偏移模型
    dataSize = locs.shape[0]
    nSample = 2
    aryLogCNK = CalLogCNK(dataSize)
    aryLogCKSample = CalLogCKSample(nSample, dataSize)
    locErr = [None] * dataSize

    # orsa 初始化
    loge0 = log10(dataSize-nSample)
    thr = sys.maxsize
    minNFA = ErrorIndex(np.array([sys.maxsize]), 100)
    res_pt = np.array([-1, -1])

    directions = compute_directions(locs, prd_directs)
    if ray_fit:
        loc_pts, s_inds = generate_models2(locs, directions) # 射线相交计算模型
    else:
        loc_pts, s_inds = generate_models(locs, directions) # 直线相交计算模型
    max_dist = 0
    inlierIdx = np.full(dataSize, False)
    sloc_errs = np.zeros(dataSize)
    best_i = -1
    for i, pt in enumerate(loc_pts):
        if pt[0] == -1:
            continue
        errs = rotation_error(pt, locs, prd_directs) # 计算距离
        for ind, data in enumerate(errs):
            locErr[ind] = ErrorIndex(data, ind)
        if reset_err:
            locErr[s_inds[i,0]].err=0
            locErr[s_inds[i, 1]].err = 0
        locErr = sorted(locErr)
        eliminate_zeros(locErr, nSample)

        logSrch = log10(140)
        # best = FindBest_theta_uniform(locErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample)
        # best = FindBest_theta_normal(locErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample)
        best = FindBest_theta(locErr, thr, loge0, aryLogCNK, aryLogCKSample, h0)
        if best.err < minNFA.err:
            minNFA.err = best.err
            minNFA.ind = best.ind
            best_i = i
            max_dist = errs[best.ind-1]
            inlierIdx = np.full(dataSize, False)
            sloc_errs = np.zeros(dataSize)
            res_pt = pt
            for j in range(best.ind):
                inlierIdx[locErr[j].ind] = True
            for kk in range(len(locErr)):
                sloc_errs[locErr[kk].ind] = locErr[kk].err
        # 绘制定位示意图
        # 绘制定位点，绘制12根射线
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255

        # B G R
        for tt, loc in enumerate(locs):
            lx, ly = loc
            dx, dy = directions[tt]
            ex_, ey_ = lx + 15 * dx, ly + 15 * dy
            if inlierIdx[tt] == True:
                # 内点为绿色
                image = cv2.line(image, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 255, 0), thickness=3)
                # 否则为红色
            else:
                image = cv2.line(image, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 0, 255), thickness=3)
            image = cv2.circle(image, center=(int(loc[0]), int(loc[1])), radius=3,
                              thickness=1,
                              color=(255, 0, 0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, str(tt), (int(loc[0]), int(loc[1]+10)), font, 0.3, (0, 0, 0), 1)
        image = cv2.circle(image, center=(int(pt[0]), int(pt[1])), radius=3,
                          thickness=-1,
                          color=(255, 0, 0))
        pass
    # 模型都不可用
    if minNFA.err == sys.maxsize:
        res_pt = np.mean(locs, axis=0)
        sloc_errs = pt2slice_rays(res_pt, locs, directions) # 计算距离
    in_num = np.sum(inlierIdx)
    in_locs = locs[inlierIdx]
    in_dires = directions[inlierIdx]
    if in_num > 2:
        res_pt, info = find_opt_pt(res_pt, in_locs, in_dires)
    if out_nfa:
        return res_pt, inlierIdx, minNFA.err, sloc_errs
    else:
        return res_pt, inlierIdx


# 使用角度theta进行判断
def location_camera_center_NFA2(locs,
                                src_area=None,
                                nfa_thr=-1,
                                out_nfa=False,
                                reset_err=True,
                                prd_directs=None,
                                ray_fit=True,
                                h0=None):

    # 遍历采样，拟合偏移模型
    dataSize = locs.shape[0]
    nSample = 2
    aryLogCNK = CalLogCNK(dataSize)
    aryLogCKSample = CalLogCKSample(nSample, dataSize)
    locErr = [None] * dataSize

    # orsa 初始化
    loge0 = log10(dataSize-nSample)
    if src_area is None:
        src_area = estimate_search_area(locs)

    logSrch = log10(150)
    thr = sys.maxsize
    minNFA = ErrorIndex(np.array([sys.maxsize]), 100)
    res_pt = np.array([-1, -1])

    directions = compute_directions(locs, prd_directs)
    if ray_fit:
        loc_pts, s_inds = generate_models2(locs, directions) # 射线相交计算模型
    else:
        loc_pts, s_inds = generate_models(locs, directions) # 直线相交计算模型
    max_dist = 0
    inlierIdx = np.full(dataSize, False)
    sloc_errs = np.zeros(dataSize)
    best_i = -1
    for i, pt in enumerate(loc_pts):
        if i == 26:
            tt_k = 1
            pass
        if pt[0] == -1:
            continue
        errs = rotation_error(pt, locs, prd_directs) # 计算距离
        for ind, data in enumerate(errs):
            locErr[ind] = ErrorIndex(data, ind)
        if reset_err:
            locErr[s_inds[i,0]].err=0
            locErr[s_inds[i, 1]].err = 0
        locErr = sorted(locErr)
        eliminate_zeros(locErr, nSample)
        # best_t = FindBest_theta_uniform(locErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample)
        # best = FindBest_theta_normal(locErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample)
        best = FindBest_theta(locErr, thr, loge0, aryLogCNK, aryLogCKSample, h0)
        if best.err < minNFA.err:
            minNFA.err = best.err
            minNFA.ind = best.ind
            best_i = i
            max_dist = errs[best.ind-1]
            inlierIdx = np.full(dataSize, False)
            sloc_errs = np.zeros(dataSize)
            res_pt = pt
            for j in range(best.ind):
                inlierIdx[locErr[j].ind] = True
            for kk in range(len(locErr)):
                sloc_errs[locErr[kk].ind] = locErr[kk].err

        # # 绘制定位示意图
        # # 绘制定位点，绘制12根射线
        # image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        # # B G R
        # for tt, loc in enumerate(locs):
        #     lx, ly = loc
        #     dx, dy = directions[tt]
        #     ex_, ey_ = lx + 15 * dx, ly + 15 * dy
        #     if inlierIdx[tt] == True:
        #         # 内点为绿色
        #         image = cv2.line(image, pt1=(int(lx), int(ly)),
        #                         pt2=(int(ex_), int(ey_)),
        #                         color=(0, 255, 0), thickness=3)
        #         # 否则为红色
        #     else:
        #         image = cv2.line(image, pt1=(int(lx), int(ly)),
        #                         pt2=(int(ex_), int(ey_)),
        #                         color=(0, 0, 255), thickness=3)
        #     image = cv2.circle(image, center=(int(loc[0]), int(loc[1])), radius=3,
        #                       thickness=1,
        #                       color=(255, 0, 0))
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(image, str(tt), (int(loc[0]), int(loc[1]+10)), font, 0.3, (0, 0, 0), 1)
        # image = cv2.circle(image, center=(int(pt[0]), int(pt[1])), radius=3,
        #                   thickness=-1,
        #                   color=(255, 0, 0))
        # pass
    # 模型都不可用
    if minNFA.err == sys.maxsize:
        res_pt = np.mean(locs, axis=0)
        sloc_errs = pt2slice_rays(res_pt, locs, directions) # 计算距离
    in_num = np.sum(inlierIdx)
    in_locs = locs[inlierIdx]
    in_dires = directions[inlierIdx]
    if in_num > 2:
        res_pt, info = find_opt_pt(res_pt, in_locs, in_dires)
    if out_nfa:
        return res_pt, inlierIdx, minNFA.err < nfa_thr, minNFA.err, sloc_errs
    else:
        return res_pt, inlierIdx, minNFA.err < nfa_thr

# 使用两种概率进行判断 , 使用各切片图像自己的概率
def location_camera_center_NFA3(locs, src_area=None, nfa_thr=-1, out_nfa=False, reset_err=True, prd_directs=None, ray_fit=True):
    # 遍历采样，拟合偏移模型
    dataSize = locs.shape[0]
    nSample = 2
    aryLogCNK = CalLogCNK(dataSize)
    aryLogCKSample = CalLogCKSample(nSample, dataSize)
    locErr = [None] * dataSize

    # orsa 初始化
    loge0 = log10(dataSize-nSample)
    if src_area is None:
        src_area = estimate_search_area(locs)

    logSrch = log10(45)+log10(5)
    thr = sys.maxsize
    minNFA = ErrorIndex(np.array([sys.maxsize]), 100)
    res_pt = np.array([-1, -1])

    directions = compute_directions(locs, prd_directs)

    loc_pts, s_inds, cam_oris = generate_models3(locs, prd_directs) # 射线相交计算模型

    max_dist = 0
    inlierIdx = np.full(dataSize, False)
    sloc_errs = np.zeros(dataSize)
    for i, pt in enumerate(loc_pts):
        if pt[0] == -1:
            continue
        errs, t_errs = rotation_error2(pt, locs, cam_oris[i], prd_directs) # 计算距离
        for ind, data in enumerate(errs):
            locErr[ind] = ErrorIndex(data, ind)
        if reset_err:
            locErr[s_inds[i,0]].err=0
            locErr[s_inds[i, 1]].err = 0
        locErr = sorted(locErr)
        eliminate_zeros(locErr, nSample)
        best = FindBest_theta(locErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample)
        if best.err < minNFA.err:
            minNFA.err = best.err
            minNFA.ind = best.ind
            max_dist = errs[best.ind-1]
            inlierIdx = np.full(dataSize, False)
            sloc_errs = np.zeros(dataSize)
            res_pt = pt
            for j in range(best.ind):
                inlierIdx[locErr[j].ind] = True
            for kk in range(len(locErr)):
                sloc_errs[locErr[kk].ind] = locErr[kk].err
        # 绘制定位示意图
        # 绘制定位点，绘制12根射线
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255

        # B G R
        for tt, loc in enumerate(locs):
            lx, ly = loc

            dx, dy = directions[tt]
            ex_, ey_ = lx + 15 * dx, ly + 15 * dy
            if inlierIdx[tt] == True:
                # 内点为绿色
                image = cv2.line(image, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 255, 0), thickness=3)
                # 否则为红色
            else:
                image = cv2.line(image, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 0, 255), thickness=3)
            image = cv2.circle(image, center=(int(loc[0]), int(loc[1])), radius=3,
                              thickness=1,
                              color=(255, 0, 0))
        image = cv2.circle(image, center=(int(pt[0]), int(pt[1])), radius=3,
                          thickness=-1,
                          color=(255, 0, 0))
        pass
    # 模型都不可用
    if minNFA.err == sys.maxsize:
        res_pt = np.mean(locs, axis=0)
        sloc_errs = pt2slice_rays(res_pt, locs, directions) # 计算距离
    in_num = np.sum(inlierIdx)
    in_locs = locs[inlierIdx]
    in_dires = directions[inlierIdx]
    if in_num > 2:
        res_pt, info = find_opt_pt(res_pt, in_locs, in_dires)
    if out_nfa:
        return res_pt, inlierIdx, minNFA.err < nfa_thr, minNFA.err, sloc_errs
    else:
        return res_pt, inlierIdx, minNFA.err < nfa_thr

# 使用两种概率进行判断 , 相机的旋转使用统一旋转
def location_camera_center_NFA4(locs, src_area=None, nfa_thr=-1, out_nfa=False, reset_err=True, prd_directs=None, ray_fit=True):
    # 遍历采样，拟合偏移模型
    dataSize = locs.shape[0]
    nSample = 2
    aryLogCNK = CalLogCNK(dataSize)
    aryLogCKSample = CalLogCKSample(nSample, dataSize)
    locErr = [None] * dataSize

    # orsa 初始化
    loge0 = log10(dataSize-nSample)
    if src_area is None:
        src_area = estimate_search_area(locs)

    logSrch = log10(10)+log10(10)
    thr = sys.maxsize
    minNFA = ErrorIndex(np.array([sys.maxsize]), 100)
    res_pt = np.array([-1, -1])

    directions = compute_directions(locs, prd_directs)
    if ray_fit:
        loc_pts, s_inds = generate_models(locs, directions)  # 直线相交计算模型
    else:
        loc_pts, s_inds = generate_models2(locs, directions)  # 射线相交计算模型

    max_dist = 0
    inlierIdx = np.full(dataSize, False)
    sloc_errs = np.zeros(dataSize)
    for i, pt in enumerate(loc_pts):
        if pt[0] == -1:
            continue
        errs, t_errs = rotation_error3(pt, locs, prd_directs) # 计算距离
        for ind, data in enumerate(errs):
            locErr[ind] = ErrorIndex(data, ind)
        if reset_err:
            locErr[s_inds[i,0]].err=0
            locErr[s_inds[i, 1]].err = 0
        locErr = sorted(locErr)
        eliminate_zeros(locErr, nSample)
        best = FindBest_theta(locErr, thr, loge0, logSrch, aryLogCNK, aryLogCKSample)
        if best.err < minNFA.err:
            minNFA.err = best.err
            minNFA.ind = best.ind
            max_dist = errs[best.ind-1]
            inlierIdx = np.full(dataSize, False)
            sloc_errs = np.zeros(dataSize)
            res_pt = pt
            for j in range(best.ind):
                inlierIdx[locErr[j].ind] = True
            for kk in range(len(locErr)):
                sloc_errs[locErr[kk].ind] = locErr[kk].err
        # 绘制定位示意图
        # 绘制定位点，绘制12根射线
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255

        # B G R
        for tt, loc in enumerate(locs):
            lx, ly = loc

            dx, dy = directions[tt]
            ex_, ey_ = lx + 15 * dx, ly + 15 * dy
            if inlierIdx[tt] == True:
                # 内点为绿色
                image = cv2.line(image, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 255, 0), thickness=3)
                # 否则为红色
            else:
                image = cv2.line(image, pt1=(int(lx), int(ly)),
                                pt2=(int(ex_), int(ey_)),
                                color=(0, 0, 255), thickness=3)
            image = cv2.circle(image, center=(int(loc[0]), int(loc[1])), radius=3,
                              thickness=1,
                              color=(255, 0, 0))
        image = cv2.circle(image, center=(int(pt[0]), int(pt[1])), radius=3,
                          thickness=-1,
                          color=(255, 0, 0))
        pass
    # 模型都不可用
    if minNFA.err == sys.maxsize:
        res_pt = np.mean(locs, axis=0)
        sloc_errs = pt2slice_rays(res_pt, locs, directions) # 计算距离
    in_num = np.sum(inlierIdx)
    in_locs = locs[inlierIdx]
    in_dires = directions[inlierIdx]
    if in_num > 2:
        res_pt, info = find_opt_pt(res_pt, in_locs, in_dires)
    if out_nfa:
        return res_pt, inlierIdx, minNFA.err < nfa_thr, minNFA.err, sloc_errs
    else:
        return res_pt, inlierIdx, minNFA.err < nfa_thr


# 计算输入的点到各射线的距离
def compute_distance(gt_pt, locs):
    directions = compute_directions(locs)
    errs = pt2slice_rays(gt_pt, locs, directions)  # 计算距离
    return errs

# 统计定位结果
def statistics_loc_res(locs):
    locs = np.array(locs)
    loc_mean2 = np.mean(locs, axis=0)
    loc_meadian2 = np.median(locs, axis=0)
    print('loc mean, median, max: ', loc_mean2, loc_meadian2)

# 统计判定结果
def statistics_valid_res(prd_valids, gt_valids):
    prd_valids = np.array(prd_valids).squeeze()
    gt_valids = np.array(gt_valids).squeeze()
    TP = np.sum((gt_valids == True) & (prd_valids == True))
    FP = np.sum((gt_valids == False) & (prd_valids == True))
    TN = np.sum((gt_valids == False) & (prd_valids == False))
    FN = np.sum((gt_valids == True) & (prd_valids == False))
    print('TP, FP, TN, FN:', TP, FP, TN, FN)
    RoTN = TN/(TN+FP)# recall of TN
    print('Recall of TN:', RoTN)
    if TN>0 or FN>0:
        PoTN = TN/(TN+FN)
        print('precision of TN:', PoTN) # precision
    else:
        PoTN = 0
        print('precision: - ')
    if PoTN == 0 and RoTN == 0:
        print('the F1: -')
    else:
        print('the F1 of TN: ', 2*PoTN*RoTN/(PoTN+RoTN))

    print('precision of TP:', TP / (TP + FP))
    print('Acc:', (TP+TN)/len(gt_valids))





