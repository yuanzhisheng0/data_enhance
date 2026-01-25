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
)

def parse_xml(xml_file):
    """解析XML标注文件"""
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

def apply_roi_augmentation(roi):
    """
    对单个目标ROI应用数据增强
    
    Args:
        roi: 目标ROI图像
    
    Returns:
        augmented_roi: 增强后的ROI
        augmentation_type: 应用的增强类型
    """
    augmentation_funcs = [shuiping, chuizhi, suiji_xuanzhuan]
    
    # 随机选择一种增强操作
    augmentation_func = random.choice(augmentation_funcs)
    
    try:
        # 为ROI创建虚拟边界框（整个ROI区域）
        h, w = roi.shape[:2]
        dummy_bboxes = [[0, 0, w, h]]

        # 应用增强函数
        augmented_roi, _ = augmentation_func(roi, dummy_bboxes)
        return augmented_roi, augmentation_func.__name__
    except Exception as e:
        # 如果增强失败，返回原ROI
        print(f"警告：ROI增强失败：{e}")
        return roi, "original"

def paste_target_back_original_background(img, bboxes, output_size=(1120, 560), num_variations=2):
    """
    将目标切割出来，进行数据增强，然后拼接回原背景
    
    Args:
        img: 原始图像
        bboxes: 目标边界框列表
        output_size: 输出图像尺寸
        num_variations: 生成的新位置图片数量
    
    Returns:
        results: 结果图像列表，包含原图像和增强后的图像
    """
    results = []
    
    # 1. 添加原图像
    results.append(img.copy())
    
    # 2. 切割目标并进行增强
    targets = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        
        # 确保边界框在图像范围内
        xmin = max(0, min(xmin, img.shape[1] - 1))
        ymin = max(0, min(ymin, img.shape[0] - 1))
        xmax = max(xmin + 1, min(xmax, img.shape[1]))
        ymax = max(ymin + 1, min(ymax, img.shape[0]))
        
        # 提取目标区域
        target_roi = img[ymin:ymax, xmin:xmax].copy()
        
        if target_roi.size == 0:
            continue
        
        # 提取红色目标区域，创建透明背景
        target_roi_bgra, red_mask = extract_red_target_with_transparency(target_roi)
        if target_roi_bgra is None:
            continue
        
        # 将BGRA转换回BGR用于增强
        target_roi_bgr = cv2.cvtColor(target_roi_bgra, cv2.COLOR_BGRA2BGR)
        targets.append({
            'roi': target_roi_bgr,
            'original_bbox': bbox
        })
    
    # 3. 生成新位置的图像
    for i in range(num_variations):
        # 创建原图像的副本作为背景
        new_img = img.copy()
        
        # 对每个目标进行增强并粘贴到新位置
        for target in targets:
            # 应用数据增强
            augmented_roi, _ = apply_roi_augmentation(target['roi'])
            
            # 提取红色目标区域，创建透明背景
            augmented_roi_bgra, red_mask = extract_red_target_with_transparency(augmented_roi)
            if augmented_roi_bgra is None:
                continue
            
            # 随机生成新位置（确保目标完全在图像范围内）
            h, w = augmented_roi_bgra.shape[:2]
            max_x = img.shape[1] - w
            max_y = img.shape[0] - h
            
            if max_x <= 0 or max_y <= 0:
                continue
            
            new_x = random.randint(0, max_x)
            new_y = random.randint(0, max_y)
            
            # 计算粘贴区域
            end_x = new_x + w
            end_y = new_y + h
            
            # 提取alpha通道
            alpha_channel = augmented_roi_bgra[:, :, 3] / 255.0
            target_bgr = augmented_roi_bgra[:, :, :3]
            
            # 使用alpha通道进行透明混合
            roi_area = new_img[new_y:end_y, new_x:end_x]
            alpha_3ch = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=2)
            blended_roi = (target_bgr * alpha_3ch + roi_area * (1 - alpha_3ch)).astype(np.uint8)
            new_img[new_y:end_y, new_x:end_x] = blended_roi
        
        results.append(new_img)
    
    return results

def process_single_image(img_path, xml_path, output_dir, img_idx):
    """
    处理单个图像，生成增强后的图像
    
    Args:
        img_path: 图像路径
        xml_path: XML标注路径
        output_dir: 输出目录
        img_idx: 图像索引
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图像 {img_path}")
        return
    
    # 解析XML标注
    try:
        _, bboxes, names, poses = parse_xml(xml_path)
        if not bboxes:
            print(f"警告：图像 {img_path} 没有目标标注")
            return
    except Exception as e:
        print(f"警告：解析XML {xml_path} 失败：{e}")
        return
    
    # 处理图像，生成增强后的图像
    try:
        results = paste_target_back_original_background(img, bboxes)
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        for i, result_img in enumerate(results):
            if i == 0:
                # 原图像
                output_filename = f"{base_name}_original.jpg"
            else:
                # 增强后的图像
                output_filename = f"{base_name}_augmented_{i}.jpg"
            
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, result_img)
            print(f"保存：{output_path}")
    except Exception as e:
        print(f"错误：处理图像 {img_path} 失败：{e}")

def main():
    """
    主函数
    """
    # 设置路径参数
    target_dir = "target"
    output_dir = "target_enhanced"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    img_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        img_files.extend(glob.glob(os.path.join(target_dir, ext)))
    
    print(f"找到 {len(img_files)} 张图像")
    
    # 处理每张图像
    for idx, img_path in enumerate(img_files):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(target_dir, f"{base_name}.xml")
        
        if not os.path.exists(xml_path):
            print(f"警告：找不到XML标注文件 {xml_path}")
            continue
        
        print(f"处理第 {idx+1}/{len(img_files)} 张图像：{img_path}")
        process_single_image(img_path, xml_path, output_dir, idx)
    
    print("\n处理完成！")
    print(f"输出目录：{output_dir}")

if __name__ == '__main__':
    main()
