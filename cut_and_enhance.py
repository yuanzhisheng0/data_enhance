#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import glob
import sys

# 添加data_enhance目录到路径，以便导入增强函数
sys.path.append(os.path.join(os.path.dirname(__file__), 'other_enhance'))

# 导入data_enhance中的增强函数
from other_enhance.enhance import (
    shuiping,  # 水平翻转
    chuizhi,  # 垂直翻转
    suiji_xuanzhuan,  # 随机角度旋转（0-360度）
    mubiao_suofang,  # 目标缩放
)


def extract_red_target_with_transparency(roi):
    """
    从ROI中提取红色目标区域，将其他部分设为透明
    
    Args:
        roi: 输入ROI图像（BGR格式）
    
    Returns:
        roi_with_alpha: 带alpha通道的图像（BGRA格式），红色区域不透明，其他区域透明
        red_mask: 红色区域的掩码
    """
    if roi.size == 0:
        return None, None

    # 转换为HSV颜色空间，便于识别红色
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 定义红色范围（OpenCV H: 0-180）
    # 红色在HSV中跨越0度和180度附近
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 形态学操作，去除噪声，填充小洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # 如果红色区域太小，返回None
    if np.sum(red_mask) < 100:
        return None, None

    # 创建BGRA图像（带alpha通道）
    roi_bgra = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)

    # 将非红色区域设为透明（alpha=0）
    # 红色区域保持不透明（alpha=255）
    roi_bgra[:, :, 3] = red_mask

    # 对alpha通道进行轻微羽化，使边缘更自然
    alpha_channel = roi_bgra[:, :, 3]
    feathered_alpha = cv2.GaussianBlur(alpha_channel, (3, 3), 0)
    roi_bgra[:, :, 3] = feathered_alpha

    return roi_bgra, red_mask


def simulate_sonar_scaled(roi_bgra, paste_x, paste_y, bg_center, max_range_m, output_size):
    """
    针对0.3倍缩放后的目标进行声呐仿真。
    特点：点稀疏、保留形态、固定半径2像素。
    原理：极坐标筛网过滤法。
    """
    if roi_bgra is None or roi_bgra.size == 0:
        return roi_bgra

    h, w = roi_bgra.shape[:2]
    # 提取原始 Alpha 通道 (作为形态基准)
    original_alpha = roi_bgra[:, :, 3]

    # 1. 提取所有可见像素坐标
    # 使用较低的阈值，确保捕捉到缩放后可能变淡的边缘
    vis_y, vis_x = np.where(original_alpha > 5)
    if vis_y.size == 0:
        return roi_bgra

    # 2. 坐标转换 (像素 -> 全局 -> 极坐标)
    cx, cy = bg_center
    global_x = paste_x + vis_x
    global_y = paste_y + vis_y
    dx = global_x - cx
    dy = global_y - cy
    # 使用 float32 提高计算精度
    r = np.sqrt(dx.astype(np.float32) ** 2 + dy.astype(np.float32) ** 2)
    theta = np.arctan2(dy, dx)

    # 3. 极坐标网格量化 (核心降采样步骤)
    # -----------------------------------------------------
    pixel_to_meter = max_range_m / float(output_size[1])

    # 【关键参数调整】：针对 0.3x 缩放的小目标
    # 目标很小，如果网格太大，整个目标可能就只剩1个点了。
    # 如果网格太小，半径为2的点就会粘连。

    # A. 径向分辨率 (Range Resolution)
    # 设为 0.5米左右。对于缩放后的目标，这能保留纵向的几个层次。
    range_res_m = 0.65
    range_step_px = range_res_m / pixel_to_meter

    # B. 方位分辨率 (Azimuth Resolution) - 最关键!
    # 为了防止半径为2 (直径4) 的点横向粘连，相邻网格中心的间距必须大于4像素。
    # 建议设置较大的角度，确保稀疏。
    # 1.5度到 2.0度是一个比较安全的值，能保证点分开。
    bearing_res_deg = 2.5
    bearing_step_rad = np.deg2rad(bearing_res_deg)

    # 量化与去重 (找出稀疏的中心点)
    r_bin = (r / range_step_px).astype(np.int32)
    theta_bin = (theta / bearing_step_rad).astype(np.int32)
    bins = np.stack((r_bin, theta_bin), axis=1)
    # return_index=True 返回 unique 元素在原数组中的索引
    _, unique_indices = np.unique(bins, axis=0, return_index=True)

    # 获取保留点的局部坐标
    kept_x = vis_x[unique_indices]
    kept_y = vis_y[unique_indices]

    # 4. 生成固定半径的“筛网”遮罩 (The Sieve Mask)
    # -----------------------------------------------------
    # 创建一个全黑的遮罩层
    sieve_mask = np.zeros((h, w), dtype=np.uint8)

    # 硬编码要求的半径
    fixed_radius = 2

    for lx, ly in zip(kept_x, kept_y):
        # 在遮罩上画实心白圆 (打孔)
        # 注意：这里画的是锐利的圆，没有模糊
        cv2.circle(sieve_mask, (lx, ly), fixed_radius, 255, -1)

    # 5. 组合最终图像 (求交集)
    # -----------------------------------------------------
    # 创建一个新的图像容器
    new_roi = np.zeros_like(roi_bgra)

    # 设置目标颜色 (例如纯红，或者保留原图颜色)
    # 如果要保留原图纹理，取消下面三行的注释，并注释掉设置纯色的代码
    # new_roi[:, :, 0] = roi_bgra[:, :, 0]
    # new_roi[:, :, 1] = roi_bgra[:, :, 1]
    # new_roi[:, :, 2] = roi_bgra[:, :, 2]
    # 这里设为纯红以便观察效果：
    new_roi[:, :, 0] = 0  # B
    new_roi[:, :, 1] = 0  # G
    new_roi[:, :, 2] = 255  # R

    # 【核心操作】：计算最终的 Alpha 通道
    # 逻辑：(原图有像素) AND (筛网有孔)
    # 结果：既保留了原图的形态边界，又被限制在了半径2的圆孔内。
    final_alpha = cv2.bitwise_and(original_alpha, sieve_mask)

    new_roi[:, :, 3] = final_alpha

    return new_roi


def parse_xml(xml_file):
    """解析XML标注文件（支持AUV格式）"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    names = []
    poses = []

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bboxes.append([xmin, ymin, xmax, ymax])

        name_elem = obj.find('name')
        if name_elem is not None:
            names.append(name_elem.text)
        else:
            names.append('target')

        pose_elem = obj.find('pose')
        if pose_elem is not None:
            poses.append(pose_elem.text)
        else:
            poses.append('Unspecified')

    return tree, bboxes, names, poses


def update_xml(tree, new_bboxes, save_path):
    """更新XML标注文件（保持AUV格式）"""
    root = tree.getroot()

    # 确保XML结构符合AUV格式
    if root.find('folder') is None:
        folder = ET.SubElement(root, 'folder')
        folder.text = 'my-project-name'

    # 更新filename标签，确保与保存的XML文件名对应
    if root.find('filename') is None:
        filename = ET.SubElement(root, 'filename')
    else:
        filename = root.find('filename')
    filename.text = os.path.basename(save_path).replace('.xml', '.png')

    # 更新path标签，确保与新的文件名对应
    if root.find('path') is None:
        path = ET.SubElement(root, 'path')
    else:
        path = root.find('path')
    path.text = f'/my-project-name/{os.path.basename(save_path).replace(".xml", ".png")}'

    if root.find('source') is None:
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unspecified'

    # 更新size信息
    size = root.find('size')
    if size is not None:
        # 从图像文件获取实际尺寸
        img_path = save_path.replace('.xml', '.png')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                size.find('width').text = str(width)
                size.find('height').text = str(height)
                size.find('depth').text = '3'

    # 更新object信息
    for i, obj in enumerate(root.findall('object')):
        if i < len(new_bboxes):
            # 确保pose标签存在
            if obj.find('pose') is None:
                ET.SubElement(obj, 'pose').text = 'Unspecified'

            # 确保truncated标签存在
            if obj.find('truncated') is None:
                ET.SubElement(obj, 'truncated').text = '0'

            # 确保difficult标签存在
            if obj.find('difficult') is None:
                ET.SubElement(obj, 'difficult').text = '0'

            # 更新边界框
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin, ymin, xmax, ymax = new_bboxes[i]
                bbox.find('xmin').text = str(max(0, xmin))
                bbox.find('ymin').text = str(max(0, ymin))
                bbox.find('xmax').text = str(max(0, xmax))
                bbox.find('ymax').text = str(max(0, ymax))

    tree.write(save_path)


def apply_roi_augmentation(roi, augmentation_funcs):
    """
    对单个目标ROI应用数据增强
    
    该函数复用 other_enhance/enhance.py 中定义的增强函数。
    对于单个ROI，将其视为完整图像，创建虚拟边界框后应用增强。
    
    Args:
        roi: 目标ROI图像
        augmentation_funcs: 可用的增强函数列表（来自 other_enhance/enhance.py）
    
    Returns:
        augmented_roi: 增强后的ROI
    """
    if not augmentation_funcs or roi.size == 0:
        return roi

    # 随机选择1-2个增强操作
    num_ops = random.randint(1, min(2, len(augmentation_funcs)))
    selected_ops = random.sample(augmentation_funcs, num_ops)

    augmented_roi = roi.copy()

    for op_func in selected_ops:
        try:
            # 为ROI创建虚拟边界框（整个ROI区域）
            # 这样可以直接复用 enhance.py 中的函数，它们期望 (img, bboxes) 参数
            h, w = augmented_roi.shape[:2]
            dummy_bboxes = [[0, 0, w, h]]

            # 应用增强函数（复用 enhance.py 中的函数）
            augmented_roi, _ = op_func(augmented_roi, dummy_bboxes)
        except Exception as e:
            # 如果增强失败，继续使用原ROI
            print(f"警告：ROI增强失败：{e}")
            continue

    return augmented_roi


def mubiao_qiege_hecheng_multi(sources, bg_img_path, output_size=(1120, 560), max_targets=3, allow_repeat=True,
                               roi_augmentation_funcs=None, bg_scale=None, bg_range=None):
    """
    从多张源图中随机选择多个目标（可重复），贴到同一张背景图上，生成多目标合成图。

    Args:
        sources: 列表，每项为 {"img": np.ndarray, "bboxes": List[[xmin,ymin,xmax,ymax]], "names": List[str], "poses": List[str]}
        bg_img_path: 背景图像路径
        output_size: 输出图像尺寸 (width, height)
        max_targets: 合成图最多贴入的目标数量
        allow_repeat: 是否允许同一目标重复被采样
        roi_augmentation_funcs: 可选的ROI增强函数列表，如果提供，会在拼接前对目标进行增强

    Returns:
        new_img: 合成后的图像
        new_bboxes: 新的边界框坐标（在输出图像坐标系下）
        new_names: 新的目标名称列表
        new_poses: 新的目标姿态列表
    """
    if not sources:
        raise ValueError("sources 为空，无法进行合成")

    bg_img = cv2.imread(bg_img_path)
    if bg_img is None:
        raise ValueError(f"无法读取背景图像 {bg_img_path}")
    bg_img = cv2.resize(bg_img, output_size, interpolation=cv2.INTER_LINEAR)

    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    _, sector_mask_u8 = cv2.threshold(bg_gray, 10, 255, cv2.THRESH_BINARY)
    sector_mask_bool = sector_mask_u8 > 0

    new_img = bg_img.copy()
    new_bboxes = []
    new_names = []
    new_poses = []

    num_targets = max_targets if max_targets is not None else 0
    attempts_per_target = 50

    flat_candidates = None
    if not allow_repeat:
        flat_candidates = []
        for src in sources:
            for i, bb in enumerate(src["bboxes"]):
                pose = src["poses"][i] if i < len(src["poses"]) else "Unspecified"
                flat_candidates.append((src["img"], bb, src["names"][i], pose))
        random.shuffle(flat_candidates)

    for _ in range(num_targets):
        chosen_img = None
        chosen_bbox = None
        chosen_name = None
        chosen_pose = None

        if allow_repeat:
            for _try in range(100):
                src = random.choice(sources)
                if not src["bboxes"]:
                    continue
                chosen_img = src["img"]
                idx = random.randrange(len(src["bboxes"]))
                chosen_bbox = src["bboxes"][idx].copy()
                chosen_name = src["names"][idx]
                chosen_pose = src["poses"][idx] if idx < len(src["poses"]) else "Unspecified"
                break
            if chosen_img is None:
                continue
        else:
            if not flat_candidates:
                break
            chosen_img, chosen_bbox, chosen_name, chosen_pose = flat_candidates.pop()
            chosen_bbox = chosen_bbox.copy()

        xmin, ymin, xmax, ymax = chosen_bbox
        xmin = max(0, min(xmin, chosen_img.shape[1] - 1))
        ymin = max(0, min(ymin, chosen_img.shape[0] - 1))
        xmax = max(xmin + 1, min(xmax, chosen_img.shape[1]))
        ymax = max(ymin + 1, min(ymax, chosen_img.shape[0]))

        target_roi = chosen_img[ymin:ymax, xmin:xmax].copy()
        if target_roi.size == 0:
            continue

        # 在拼接前对目标ROI进行数据增强（如果提供了增强函数）
        if roi_augmentation_funcs:
            target_roi = apply_roi_augmentation(target_roi, roi_augmentation_funcs)

        # 根据背景尺度先对裁切目标进行缩放
        if bg_scale is not None:
            try:
                h0, w0 = target_roi.shape[:2]
                target_roi, _ = mubiao_suofang(target_roi, [[0, 0, w0, h0]], scale_range=bg_scale)
            except Exception:
                pass

        # 提取红色目标区域，创建透明背景（只保留红色区域）
        target_roi_bgra, red_mask = extract_red_target_with_transparency(target_roi)
        if target_roi_bgra is None:
            continue

        # 使用旋转（后续流程会旋转ROI via suiji_xuanzhuan if needed）
        # 将BGRA转换回BGR用于旋转（旋转函数需要BGR格式）
        target_roi = cv2.cvtColor(target_roi_bgra, cv2.COLOR_BGRA2BGR)

        # 对裁切目标进行随机角度旋转（0-360度）
        h_roi, w_roi = target_roi.shape[:2]
        dummy_bboxes = [[0, 0, w_roi, h_roi]]
        target_roi, _ = suiji_xuanzhuan(target_roi, dummy_bboxes, angle_range=(0, 360))

        # 旋转后重新提取红色区域并创建透明背景
        target_roi_bgra, red_mask = extract_red_target_with_transparency(target_roi)
        if target_roi_bgra is None:
            continue

        # 使用旋转后的ROI（带透明背景）和红色掩码
        rotated_roi = target_roi_bgra
        rotated_mask = red_mask

        # 对掩码进行形态学处理和羽化
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shrunk_mask = cv2.erode(rotated_mask, erode_kernel, iterations=1)
        feathered_mask = cv2.GaussianBlur(shrunk_mask, (5, 5), 0)

        target_h, target_w = rotated_roi.shape[:2]
        max_x = output_size[0] - target_w
        max_y = output_size[1] - target_h
        if max_x <= 0 or max_y <= 0:
            continue

        attempts = 0
        while attempts < 50:
            new_x = random.randint(0, max_x)
            new_y = random.randint(0, max_y)

            overlap = False
            for ex in new_bboxes:
                ex_xmin, ex_ymin, ex_xmax, ex_ymax = ex
                overlap_x1 = max(new_x, ex_xmin)
                overlap_y1 = max(new_y, ex_ymin)
                overlap_x2 = min(new_x + target_w, ex_xmax)
                overlap_y2 = min(new_y + target_h, ex_ymax)
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    target_area = target_w * target_h
                    if overlap_area > target_area * 0.3:
                        overlap = True
                        break

            if not overlap:
                sector_crop_bool = sector_mask_bool[new_y:new_y + target_h, new_x:new_x + target_w]
                if sector_crop_bool.shape[0] != target_h or sector_crop_bool.shape[1] != target_w:
                    overlap = True
                else:
                    roi_mask_bool = (rotated_mask > 0)
                    if roi_mask_bool.any() and not np.all(sector_crop_bool[roi_mask_bool]):
                        overlap = True

            if not overlap:
                break
            attempts += 1

        if attempts >= 50:
            continue
        # 将目标粘贴到新位置（使用alpha通道进行透明混合）
        end_x = min(new_x + target_w, output_size[0])
        end_y = min(new_y + target_h, output_size[1])
        actual_w = end_x - new_x
        actual_h = end_y - new_y
        if actual_w <= 0 or actual_h <= 0:
            continue

        # 调整目标区域尺寸（如果是BGRA格式）
        if rotated_roi.shape[2] == 4:  # BGRA格式
            if actual_w != target_w or actual_h != target_h:
                rotated_roi = cv2.resize(rotated_roi, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)

            # 提取alpha通道
            alpha_channel = rotated_roi[:, :, 3] / 255.0
            target_bgr = rotated_roi[:, :, :3]
        else:  # BGR格式（兼容旧代码）
            if actual_w != target_w or actual_h != target_h:
                rotated_roi = cv2.resize(rotated_roi, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)
            # 使用羽化后的掩码作为alpha通道
            mask_single = feathered_mask
            if actual_w != target_w or actual_h != target_h:
                mask_single = cv2.resize(mask_single, (actual_w, actual_h), interpolation=cv2.INTER_NEAREST)
            alpha_channel = mask_single / 255.0
            target_bgr = rotated_roi

        # 在将要进行 alpha 混合前，根据目标在背景上的位置与背景的最大距离做可选降采样
        if bg_range is not None and bg_range > 25:
            bg_center = (560, 560)
            try:
                rotated_roi = simulate_sonar_scaled(rotated_roi, new_x, new_y, bg_center, bg_range,
                                                        output_size)
                # 重新提取alpha通道和目标BGR（降采样后）
                if rotated_roi.shape[2] == 4:
                    alpha_channel = rotated_roi[:, :, 3] / 255.0
                    target_bgr = rotated_roi[:, :, :3]
                    rotated_mask = (rotated_roi[:, :, 3] > 10).astype(np.uint8) * 255
                else:
                    # 对于BGR格式，降采样可能改变了尺寸，需要重新计算
                    target_bgr = rotated_roi
            except Exception:
                pass

        # 获取背景区域
        roi_area = new_img[new_y:end_y, new_x:end_x]
        sector_crop = sector_mask_u8[new_y:end_y, new_x:end_x]

        # 调整扇形掩码尺寸（如果需要）
        if sector_crop.shape[:2] != (actual_h, actual_w):
            sector_crop = cv2.resize(sector_crop, (actual_w, actual_h), interpolation=cv2.INTER_NEAREST)

        # 将alpha通道与扇形掩码相与，确保只粘贴到有效区域
        sector_mask_float = (sector_crop / 255.0).astype(np.float32)
        final_alpha = alpha_channel * sector_mask_float

        # 使用alpha通道进行混合
        final_alpha_3ch = np.stack([final_alpha, final_alpha, final_alpha], axis=2)
        blended_roi = (target_bgr * final_alpha_3ch + roi_area * (1 - final_alpha_3ch)).astype(np.uint8)
        new_img[new_y:end_y, new_x:end_x] = blended_roi

        # 更新掩码用于边界框计算
        merged_mask = (final_alpha * 255).astype(np.uint8)

        ys, xs = np.where(merged_mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        bbox_xmin = int(new_x + xs.min())
        bbox_ymin = int(new_y + ys.min())
        bbox_xmax = int(new_x + xs.max() + 1)
        bbox_ymax = int(new_y + ys.max() + 1)
        new_bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        new_names.append(chosen_name)
        new_poses.append(chosen_pose)

    return new_img, new_bboxes, new_names, new_poses


def complete_enhancement_pipeline(source_img_dir, source_xml_dir, bg_img_path,
                                  final_output_dir, cut_move_augmentations=10,
                                  output_size=(1120, 560), images_per_bg=None):
    """
    数据增强流水线：对目标进行数据增强，然后切割并拼接到背景图（背景图保持不变）
    
    Args:
        source_img_dir: 源图像目录
        source_xml_dir: 源标注目录
        bg_img_path: 背景图像路径
        final_output_dir: 最终输出目录
        cut_move_augmentations: 每张源图像生成的增强倍数（当 images_per_bg 为 None 时使用）
        output_size: 输出图像尺寸
        images_per_bg: 如果设置，则针对该背景生成固定数量的最终图像（覆盖 cut_move_augmentations 计算方式）
    """

    # 创建最终输出目录
    final_images_dir = os.path.join(final_output_dir, "final_images")
    final_annotations_dir = os.path.join(final_output_dir, "final_annotations")

    # 创建所有必要的目录
    os.makedirs(final_images_dir, exist_ok=True)
    os.makedirs(final_annotations_dir, exist_ok=True)

    # 检查输入目录是否存在
    if not os.path.exists(source_img_dir):
        print(f"错误：源图像目录不存在 {source_img_dir}")
        return

    if not os.path.exists(source_xml_dir):
        print(f"错误：源标注目录不存在 {source_xml_dir}")
        return

    if not os.path.exists(bg_img_path):
        print(f"错误：背景图像不存在 {bg_img_path}")
        return

    # 获取所有源图像文件
    img_files = glob.glob(os.path.join(source_img_dir, "*.png"))
    img_files.extend(glob.glob(os.path.join(source_img_dir, "*.jpg")))
    img_files.extend(glob.glob(os.path.join(source_img_dir, "*.jpeg")))

    if not img_files:
        print("错误：在源目录中没有找到图像文件")
        return

    print(f"找到 {len(img_files)} 张源图像")
    print("=" * 50)
    print("目标切割和背景合成（多图拼接）")
    print("=" * 50)

    # 定义ROI增强函数（用于在拼接前对单个目标进行增强）
    # 注意：只使用适合单个ROI的增强函数
    # 直接复用 other_enhance/enhance.py 中定义的函数
    # 注意：旋转会在增强后作为独立步骤应用，所以不包含在列表中
    roi_augmentation_funcs = [
        shuiping,  # 水平翻转
        chuizhi,  # 垂直翻转
    ]

    # 预先收集全部源图 + 标注框
    sources = []
    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(source_xml_dir, f"{base_name}.xml")
        if not os.path.exists(xml_path):
            continue
        img_mat = cv2.imread(img_path)
        if img_mat is None:
            continue
        _, bboxes, names, poses = parse_xml(xml_path)
        if bboxes:
            sources.append({"img": img_mat, "bboxes": bboxes, "names": names, "poses": poses})

    if not sources:
        print("错误：没有可用于拼接的目标")
        return

    # 根据背景文件名设定目标缩放比例（以16m为基准）
    bg_base = os.path.splitext(os.path.basename(bg_img_path))[0].lower()
    if bg_base.startswith("16"):
        bg_scale = 1.0
    elif bg_base.startswith("25"):
        bg_scale = 0.8
    elif bg_base.startswith("50"):
        bg_scale = 0.5
    elif bg_base.startswith("100"):
        bg_scale = 0.3
    # 背景对应的最大距离（米），用于降采样判断
    if bg_base.startswith("16"):
        bg_range = 16
    elif bg_base.startswith("25"):
        bg_range = 25
    elif bg_base.startswith("50"):
        bg_range = 50
    elif bg_base.startswith("100"):
        bg_range = 100
    else:
        bg_range = None

    # 多图目标切割并合成
    final_count = 1
    # 允许按每个背景指定生成数量；若 images_per_bg 提供，则使用该值，否则按源图数量 * 倍数计算
    if images_per_bg is not None:
        total_to_generate = int(images_per_bg)
    else:
        total_to_generate = len(img_files) * cut_move_augmentations
    for idx in range(total_to_generate):
        try:
            dynamic_max_targets = random.randint(3, 6)
            new_img, new_bboxes, new_names, new_poses = mubiao_qiege_hecheng_multi(
                sources,
                bg_img_path,
                output_size=output_size,
                max_targets=dynamic_max_targets,
                allow_repeat=True,
                roi_augmentation_funcs=roi_augmentation_funcs,  # 在拼接前对目标进行增强
                bg_scale=bg_scale,
                bg_range=bg_range
            )

            new_img_name = f"final_{final_count:04d}.png"
            new_xml_name = f"final_{final_count:04d}.xml"

            cv2.imwrite(os.path.join(final_images_dir, new_img_name), new_img)

            # 生成匹配目标数量的VOC标注（按照AUV格式）
            root = ET.Element('annotation')

            # 添加folder标签
            folder = ET.SubElement(root, 'folder')
            folder.text = 'my-project-name'

            # 添加filename标签
            filename = ET.SubElement(root, 'filename')
            filename.text = new_img_name

            # 添加path标签
            path = ET.SubElement(root, 'path')
            path.text = f'/my-project-name/{new_img_name}'

            # 添加source标签
            source = ET.SubElement(root, 'source')
            database = ET.SubElement(source, 'database')
            database.text = 'Unspecified'

            # 添加size标签
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(output_size[0])
            ET.SubElement(size, 'height').text = str(output_size[1])
            ET.SubElement(size, 'depth').text = '3'

            # 添加object标签
            for i, bbox in enumerate(new_bboxes):
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = new_names[i] if i < len(new_names) else 'target'
                ET.SubElement(obj, 'pose').text = new_poses[i] if i < len(new_poses) else 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'
                bnd = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bnd, 'xmin').text = str(bbox[0])
                ET.SubElement(bnd, 'ymin').text = str(bbox[1])
                ET.SubElement(bnd, 'xmax').text = str(bbox[2])
                ET.SubElement(bnd, 'ymax').text = str(bbox[3])
            tree = ET.ElementTree(root)
            tree.write(os.path.join(final_annotations_dir, new_xml_name))

            print(f"  生成增强图像 {final_count}: {len(new_bboxes)} 个目标（期望 {dynamic_max_targets}）")
            final_count += 1
        except Exception as e:
            print(f"生成增强图像时出错：{e}")
            continue

    print("=" * 50)
    print("数据增强流水线完成！")
    print(f"共生成增强图像数：{final_count - 1}")
    print(f"最终输出目录：{final_images_dir}, {final_annotations_dir}")


if __name__ == '__main__':
    # 设置路径参数
    source_img_dir = "source/images"  # 源图像目录
    source_xml_dir = "source/annotations"  # 源标注目录
    backgrounds = ["16m.png", "25m.png", "50m.png", "100m.png"]  # 要使用的背景图列表
    final_output_dir = "complete_enhanced"  # 最终输出目录（每个背景会生成一个子目录）

    # 设置增强参数
    cut_move_augmentations = 10

    output_size = (1120, 560)  # 输出图像尺寸
    images_per_background = 30  # 每个背景生成的图片数量

    print(f"源图像目录：{source_img_dir}")
    print(f"源标注目录：{source_xml_dir}")
    print(f"最终输出目录：{final_output_dir}")

    # 对每个背景分别生成 images_per_background 张图片，输出到不同子目录
    for bg in backgrounds:
        bg_path = bg
        bg_name = os.path.splitext(os.path.basename(bg))[0]
        out_dir = os.path.join(final_output_dir, bg_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"开始为背景 {bg_path} 生成 {images_per_background} 张增强图，输出到 {out_dir}")
    complete_enhancement_pipeline(
        source_img_dir, source_xml_dir, bg_path, out_dir,
        cut_move_augmentations, output_size, images_per_bg=images_per_background
    )