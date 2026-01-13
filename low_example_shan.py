import cv2
import numpy as np


def polar_point_cloud_downsample(crop_img, bbox, sonar_info, downsample_rate=(2.0, 2.0)):
    """
    对声呐目标进行基于极坐标栅格的点云降采样（减少点数量）。

    Args:
        crop_img: 裁剪的目标小图 (BGR 或 RGB)
        bbox: (xmin, ymin, xmax, ymax) 在原图中的位置
        sonar_info: 字典，包含 {'center': (cx, cy), 'max_radius': r}
        downsample_rate: (r_scale, theta_scale)
        数值越大，点越稀疏。例如 (2.0, 2.0) 表示距离和角度分辨率都降低一半。
    """
    # 1. 提取所有红色点的坐标
    # 假设红色通道是最后一个 (BGR->2, RGB->0? 请根据实际读取格式调整)
    # 这里默认 OpenCV 读取为 BGR，红色在通道 2
    # 使用阈值提取强回波点
    mask = crop_img[:, :, 2] > 50
    y_idxs, x_idxs = np.where(mask)

    if len(y_idxs) == 0:
        return crop_img  # 没有点，直接返回

    # 2. 将局部坐标转换为全局坐标
    xmin, ymin, _, _ = bbox
    global_x = x_idxs + xmin
    global_y = y_idxs + ymin

    # 3. 转换为极坐标 (Range, Theta)
    cx, cy = sonar_info['center']
    dx = global_x - cx
    dy = global_y - cy  # 注意：OpenCV中y向下为正，如果原点在底部，这里通常是负值

    # 计算半径 R
    r = np.sqrt(dx ** 2 + dy ** 2)
    # 计算角度 Theta
    theta = np.arctan2(dy, dx)

    # 4. 【核心步骤】极坐标栅格量化 (Quantization)
    # 通过除以缩放因子并取整，将相邻的点归并到同一个 ID 中
    r_scale, theta_scale = downsample_rate

    # 为了量化 theta，我们需要一个基准分辨率，这里简化处理：
    # 直接对计算出的浮点 r 和 theta 进行较粗粒度的取整
    # r_scale: 比如 16m -> 50m，原本 1 pixel 代表 3cm，现在需要 1 bin 代表 10cm
    # 这里的 scale 是相对于当前像素网格的倍率

    r_quantized = np.round(r / r_scale).astype(np.int32)

    # theta 需要特殊处理，因为它是弧度。
    # 假设原本的角分辨率对应图像上的 pixel 跨度，这里我们也放大间隔
    # 简单的做法是直接对 theta 值做量化
    theta_step = 0.005 * theta_scale  # 假设一个基础弧度步长，乘以降采样倍率
    theta_quantized = np.round(theta / theta_step).astype(np.int32)

    # 5. 去重
    # 将 (r_bin, theta_bin) 组合，去除重复项
    # 这意味着原本同一个大格子里的 10 个点，现在只剩 1 个坐标
    stacked = np.stack((r_quantized, theta_quantized), axis=1)
    _, unique_indices = np.unique(stacked, axis=0, return_index=True)

    # 6. 保留筛选后的点
    kept_r = r[unique_indices]
    kept_theta = theta[unique_indices]

    # 7. 还原回笛卡尔坐标 (全局)
    new_dx = kept_r * np.cos(kept_theta)
    new_dy = kept_r * np.sin(kept_theta)

    new_global_x = (new_dx + cx).astype(np.int32)
    new_global_y = (new_dy + cy).astype(np.int32)

    # 8. 映射回局部 Crop 坐标
    new_local_x = new_global_x - xmin
    new_local_y = new_global_y - ymin

    # 9. 重绘图像
    result_img = np.zeros_like(crop_img)
    h, w = crop_img.shape[:2]

    # 过滤越界点 (因为取整和还原可能会导致轻微坐标漂移)
    valid_mask = (new_local_x >= 0) & (new_local_x < w) & \
                 (new_local_y >= 0) & (new_local_y < h)

    final_x = new_local_x[valid_mask]
    final_y = new_local_y[valid_mask]

    # 在结果图上画红点
    # 注意：剩下的点是单像素的
    # 如果想模拟声呐的“斑点感”，可以用 circle 画稍微大一点(半径1)
    result_img[final_y, final_x] = [0, 0, 255]  # BGR Red

    # 可选：如果觉得单像素太细，可以做一次轻微膨胀
    kernel = np.ones((2, 2), np.uint8)
    result_img = cv2.dilate(result_img, kernel, iterations=1)

    return result_img


# 假设参数
orig_h, orig_w = 560, 1120
sonar_center = (orig_w // 2, orig_h)  # 假设圆心在底部中心

# 加载您的图片
img_path = './image/frogman.png'  # 您的文件名
crop_img = cv2.imread(img_path)

if crop_img is not None:
    # 假设这个目标在原图的某个位置 (必须指定，否则极坐标计算不准)
    # 这里随便写一个位置用于演示
    bbox = (400, 200, 400 + crop_img.shape[1], 200 + crop_img.shape[0])

    # 运行降采样
    # downsample_rate=(3.0, 3.0) 表示现在的 3x3 个极坐标单元格合并成 1 个点
    # 这个值越大，点越少
    sparse_img = polar_point_cloud_downsample(
        crop_img,
        bbox,
        {'center': sonar_center, 'max_radius': 560},
        downsample_rate=(6.0, 6.0)
    )

    # 显示
    cv2.imshow("Original", crop_img)
    cv2.imshow("Sparse Points (No Blur)", sparse_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('polar_frogman_100_big.png', sparse_img)
