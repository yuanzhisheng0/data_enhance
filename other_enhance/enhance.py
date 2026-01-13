import os
import random

import cv2
import numpy as np
import xml.etree.ElementTree as ET

img_path = ''
xml_path = ''

output_img_path = "output/images"
output_xml_path = "output/annotations"


# 创建目录
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_xml_path, exist_ok=True)


# 1、水平翻转
def shuiping(img, bboxes):
    h, w = img.shape[:2]
    image = cv2.flip(img, 1)
    new_boxes = []
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        new_boxes.append([w - xmax, ymin, w - xmin, ymax])
    return image, new_boxes


# 2、垂直翻转
def chuizhi(img, bboxes):
    h, w = img.shape[:2]
    image = cv2.flip(img, 0)
    new_boxes = []
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        new_boxes.append([xmin, h - ymax, xmax, h - ymin])
    return image, new_boxes


# 3、随机角度旋转（支持0-360度，用于裁切目标）
def suiji_xuanzhuan(img, bboxes, angle_range=(0, 360)):
    """
    对图像进行随机角度旋转，并相应调整边界框坐标
    适用于裁切下来的目标，支持0-360度任意角度旋转
    
    Args:
        img: 输入图像
        bboxes: 边界框列表 [[xmin, ymin, xmax, ymax], ...]
        angle_range: 旋转角度范围，可以是元组 (min, max) 或单个数值，默认 (0, 360)
    
    Returns:
        rotated_img: 旋转后的图像（自适应尺寸）
        new_boxes: 调整后的边界框坐标
    """
    h, w = img.shape[:2]
    
    # 确定旋转角度
    if isinstance(angle_range, (int, float)):
        angle = float(angle_range)
    elif isinstance(angle_range, (tuple, list)) and len(angle_range) == 2:
        angle = random.uniform(angle_range[0], angle_range[1])
    else:
        angle = random.uniform(0, 360)
    
    # 计算旋转中心
    center = (w / 2.0, h / 2.0)
    
    # 计算旋转矩阵
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后图像的新尺寸（确保完整包含旋转后的图像）
    cos_a = abs(rot_mat[0, 0])
    sin_a = abs(rot_mat[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    # 调整旋转矩阵的平移部分，使旋转后的图像居中
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]
    
    # 旋转图片（使用边界复制模式，避免黑色边缘）
    rotated_img = cv2.warpAffine(
        img, rot_mat, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    # 旋转边界框
    new_boxes = []
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        
        # 计算四个角的坐标
        points = np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax]
        ], dtype=np.float32)
        
        # 将点转换为齐次坐标
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        
        # 应用旋转矩阵
        rotated_points = rot_mat.dot(points_ones.T).T
        
        # 获取新的边界框（最小外接矩形）
        xs = rotated_points[:, 0]
        ys = rotated_points[:, 1]
        new_xmin = max(0, int(np.floor(xs.min())))
        new_ymin = max(0, int(np.floor(ys.min())))
        new_xmax = min(new_w, int(np.ceil(xs.max())))
        new_ymax = min(new_h, int(np.ceil(ys.max())))
        
        # 确保边界框有效
        if new_xmax > new_xmin and new_ymax > new_ymin:
            new_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return rotated_img, new_boxes


# 5、随机遮挡
def suiji_zhedang(img, bboxes):
    """
    对目标框内进行随机遮挡，边界框坐标不变
    """
    occlude_ratio_range = (0.1, 0.3)
    occluded_img = img.copy()

    for box in bboxes:
        xmin, ymin, xmax, ymax = box

        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w <= 0 or box_h <= 0:
            continue

        # 随机选择遮挡比例
        occ_ratio = random.uniform(*occlude_ratio_range)
        occ_area = int(box_w * box_h * occ_ratio)

        # 随机选择遮挡框的宽高（保持在目标框内）
        max_occ_w = min(box_w, int(box_w * 0.8))
        max_occ_h = min(box_h, int(box_h * 0.8))
        if max_occ_w <= 1 or max_occ_h <= 1:
            continue

        occ_w = random.randint(int(max_occ_w * 0.3), max_occ_w)
        occ_h = max(1, int(occ_area / occ_w))
        occ_h = min(occ_h, max_occ_h)

        # 随机选择遮挡框左上角坐标（保证在目标框内）
        x1 = random.randint(xmin, xmax - occ_w)
        y1 = random.randint(ymin, ymax - occ_h)
        x2 = x1 + occ_w
        y2 = y1 + occ_h

        # 遮挡方式：用灰色或黑色块填充
        color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
        occluded_img[y1:y2, x1:x2] = color

    # 边界框坐标保持不变
    return occluded_img, bboxes


# 5.5、随机遮挡（用于裁切目标，红色点消失变为灰色背景）


# 6、缩放填充
def suofang_tianchong(img, bboxes):
    """
    缩放填充，边界框坐标相应调整
    """
    scale_range = (0.6, 1.4)
    target_size = (1120, 560)

    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图片
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建填充后的新图像
    pad_w = max(target_size[0] - new_w, 0)
    pad_h = max(target_size[1] - new_h, 0)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # 以黑色填充
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # 如果缩放后比目标尺寸大，则中心裁剪
    if img_padded.shape[0] > target_size[1] or img_padded.shape[1] > target_size[0]:
        start_y = (img_padded.shape[0] - target_size[1]) // 2
        start_x = (img_padded.shape[1] - target_size[0]) // 2
        img_padded = img_padded[start_y:start_y + target_size[1], start_x:start_x + target_size[0]]
        crop_x = start_x
        crop_y = start_y
    else:
        crop_x = 0
        crop_y = 0

    # 调整边界框坐标
    new_boxes = []
    for box in bboxes:
        xmin, ymin, xmax, ymax = box

        # 缩放
        xmin = int(xmin * scale)
        xmax = int(xmax * scale)
        ymin = int(ymin * scale)
        ymax = int(ymax * scale)

        # 填充偏移
        xmin = xmin + pad_left - crop_x
        xmax = xmax + pad_left - crop_x
        ymin = ymin + pad_top - crop_y
        ymax = ymax + pad_top - crop_y

        # 限制在图像范围内
        xmin = max(0, min(xmin, target_size[0] - 1))
        xmax = max(0, min(xmax, target_size[0] - 1))
        ymin = max(0, min(ymin, target_size[1] - 1))
        ymax = max(0, min(ymax, target_size[1] - 1))

        new_boxes.append([xmin, ymin, xmax, ymax])

    return img_padded, new_boxes


# 6.5、目标缩放（用于裁切后的目标）
def mubiao_suofang(img, bboxes, scale_range=(0.7, 1.3)):
    """
    对图像进行缩放，并相应调整边界框坐标
    适用于裁切下来的目标，可以控制缩放比例
    
    Args:
        img: 输入图像
        bboxes: 边界框列表 [[xmin, ymin, xmax, ymax], ...]
        scale_range: 缩放比例范围，可以是元组 (min, max) 或单个数值
    
    Returns:
        scaled_img: 缩放后的图像
        new_boxes: 调整后的边界框坐标
    """
    h, w = img.shape[:2]
    
    # 如果 scale_range 是单个数值，直接使用
    if isinstance(scale_range, (int, float)):
        scale = float(scale_range)
    # 如果是元组，随机选择范围内的值
    elif isinstance(scale_range, (tuple, list)) and len(scale_range) == 2:
        scale = random.uniform(scale_range[0], scale_range[1])
    else:
        # 默认缩放范围
        scale = random.uniform(0.7, 1.3)
    
    # 计算新尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 确保新尺寸至少为1
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    # 缩放图像
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 调整边界框坐标
    new_boxes = []
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        
        # 缩放边界框坐标
        new_xmin = int(xmin * scale)
        new_ymin = int(ymin * scale)
        new_xmax = int(xmax * scale)
        new_ymax = int(ymax * scale)
        
        # 确保坐标在图像范围内
        new_xmin = max(0, min(new_xmin, new_w - 1))
        new_ymin = max(0, min(new_ymin, new_h - 1))
        new_xmax = max(new_xmin + 1, min(new_xmax, new_w))
        new_ymax = max(new_ymin + 1, min(new_ymax, new_h))
        
        new_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return scaled_img, new_boxes


# 标注处理--------------------------------------------------
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
    return tree, bboxes


def update_xml(tree, new_bboxes, save_path):
    root = tree.getroot()
    for i, obj in enumerate(root.findall('object')):
        bbox = obj.find('bndbox')
        xmin, ymin, xmax, ymax = new_bboxes[i]
        bbox.find('xmin').text = str(max(0, xmin))
        bbox.find('ymin').text = str(max(0, ymin))
        bbox.find('xmax').text = str(max(0, xmax))
        bbox.find('ymax').text = str(max(0, ymax))
    tree.write(save_path)


# 主处理流程 --------------------------------------------------
if __name__ == '__main__':
    augment_funcs = [
        shuiping,
        chuizhi,
        suiji_xuanzhuan,
        # liangdu_duibidu,
        suiji_zhedang,
        suofang_tianchong,
        # bianhuan_yanse,
        # zaosheng_tianjia
    ]

    image_ids = list(range(1, 81))  # out1~out80
    save_count = 1

    for i in image_ids:
        img_file = os.path.join(img_path, f"out{i}.png")
        xml_file = os.path.join(xml_path, f"out{i}.xml")

        image = cv2.imread(img_file)
        tree, bboxes = parse_xml(xml_file)

        for _ in range(20):  # 每张生成20张
            img_aug = image.copy()
            boxes_aug = [b.copy() for b in bboxes]
            ops = random.sample(augment_funcs, k=random.randint(2, 4))  # 随机组合2~4个

            for op in ops:
                img_aug, boxes_aug = op(img_aug, boxes_aug)

            new_img_name = f"out{save_count}.png"
            new_xml_name = f"out{save_count}.xml"

            cv2.imwrite(os.path.join(output_img_path, new_img_name), img_aug)
            update_xml(tree, boxes_aug, os.path.join(output_xml_path, new_xml_name))

            save_count += 1

    print("全部增强完成！共生成图像数：", save_count - 1)
