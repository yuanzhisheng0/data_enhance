#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET


# ==========================================
# 1. 基础辅助函数 (XML解析与目标提取)
# ==========================================

def parse_xml(xml_path):
    """解析XML获取目标坐标和名称"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    names = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        bnd = obj.find('bndbox')
        xmin = int(float(bnd.find('xmin').text))
        ymin = int(float(bnd.find('ymin').text))
        xmax = int(float(bnd.find('xmax').text))
        ymax = int(float(bnd.find('ymax').text))
        bboxes.append([xmin, ymin, xmax, ymax])
        names.append(name)

    return tree, bboxes, names


def extract_target_with_alpha(img_roi):
    """
    将矩形目标ROI转换为带透明通道(BGRA)的图像。
    假设声呐目标背景是黑色的，将黑色背景转为透明。
    """
    if img_roi is None or img_roi.size == 0:
        return None, None

    # 转为灰度图用于生成Mask
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

    # 简单的阈值处理：非黑色区域认为是目标
    # 阈值可根据实际素材调整，一般声呐目标较亮，20-40是安全范围
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 增加一点柔和边缘
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # 分离通道并增加Alpha通道
    b, g, r = cv2.split(img_roi)
    bgra = cv2.merge([b, g, r, mask])

    return bgra, mask


# ==========================================
# 2. 核心算法：生成两墙之间的掩膜
# ==========================================

def get_region_between_walls(bg_img):
    """
    生成左右两道红色池壁“之间”的区域掩膜。
    """
    h, w = bg_img.shape[:2]
    hsv_bg = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)

    # --- A. 提取红色池壁 (HSV) ---
    # 红色在HSV中分布在 0-10 和 160-180 两个区间
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_bg, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_bg, lower_red2, upper_red2)
    wall_mask = cv2.bitwise_or(mask1, mask2)

    # 预处理：去噪，闭运算连接断裂的墙壁
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_clean)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel_clean)
    # --- B. 找到候选墙体轮廓并选择左右两墙 ---
    cnts, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(wall_mask)

    # 计算每个轮廓的面积和质心x，过滤太小的噪声轮廓
    cand = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < max(50, 0.001 * w * h):
            continue
        M = cv2.moments(c)
        if M.get('m00', 0) == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cand.append((c, area, cx, cy))

    if len(cand) < 2:
        # 如果少于两个轮廓，尽量退回使用扫描线法填充大区
        water_mask = np.zeros_like(wall_mask)
        for y in range(h):
            row_indices = np.where(wall_mask[y, :] > 0)[0]
            if len(row_indices) > 1:
                left_limit = row_indices[0]
                right_limit = row_indices[-1]
                if right_limit - left_limit > w * 0.1:
                    water_mask[y, left_limit:right_limit] = 255
        # 安全腐蚀
        erosion_val = max(3, int(w * 0.02))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_val, erosion_val))
        safe_water_mask = cv2.erode(water_mask, kernel_erode, iterations=1)
        wall_thick = cv2.dilate(wall_mask, kernel_clean, iterations=2)
        safe_water_mask = cv2.bitwise_and(safe_water_mask, cv2.bitwise_not(wall_thick))
        return safe_water_mask

    # 按质心x排序，选左右两侧最大的两个轮廓
    cand = sorted(cand, key=lambda x: x[2])
    left_candidates = [c for c in cand if c[2] < w // 2]
    right_candidates = [c for c in cand if c[2] >= w // 2]
    if left_candidates:
        left = max(left_candidates, key=lambda x: x[1])[0]
    else:
        left = min(cand, key=lambda x: x[2])[0]
    if right_candidates:
        right = max(right_candidates, key=lambda x: x[1])[0]
    else:
        right = max(cand, key=lambda x: x[2])[0]

    # 生成左右墙体mask
    left_mask = np.zeros_like(wall_mask)
    right_mask = np.zeros_like(wall_mask)
    cv2.drawContours(left_mask, [left], -1, 255, -1)
    cv2.drawContours(right_mask, [right], -1, 255, -1)

    # 对墙体做细微膨胀以覆盖完整厚度
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    left_mask = cv2.dilate(left_mask, kernel_expand, iterations=2)
    right_mask = cv2.dilate(right_mask, kernel_expand, iterations=2)

    # --- C. 在左右墙之间按列全高填充（确保高度覆盖整张图像） ---
    water_mask = np.zeros_like(wall_mask)
    # 计算左墙最右边界和右墙最左边界（全图范围）
    left_cols = np.where(np.any(left_mask > 0, axis=0))[0]
    right_cols = np.where(np.any(right_mask > 0, axis=0))[0]
    if left_cols.size == 0 or right_cols.size == 0:
        return np.zeros_like(wall_mask)
    x_left = left_cols.max()
    x_right = right_cols.min()
    # 若左右边界不合理则返回空
    if x_right <= x_left:
        return np.zeros_like(wall_mask)
    # 仅在左右墙间距足够大时填充整个列高区域
    if (x_right - x_left) > max(2, int(w * 0.02)):
        water_mask[:, x_left:x_right] = 255

    # 安全收缩，避免靠墙太近
    erosion_val = max(3, int(w * 0.02))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_val, erosion_val))
    safe_water_mask = cv2.erode(water_mask, kernel_erode, iterations=1)
    # 移除与墙体重叠区域
    wall_thick = cv2.dilate(wall_mask, kernel_clean, iterations=3)
    safe_water_mask = cv2.bitwise_and(safe_water_mask, cv2.bitwise_not(wall_thick))

    return safe_water_mask


# ==========================================
# 3. 主程序逻辑
# ==========================================

def paste_targets_between_walls(bg_path='bg_chibi.jpg',
                                source_img_dir='source/images',
                                source_xml_dir='source/annotations',
                                out_dir='complete_enhanced/chibi',
                                num_targets=None,
                                num_images=5):
    # 准备输出目录
    out_images_dir = os.path.join(out_dir, 'final_images')
    out_ann_dir = os.path.join(out_dir, 'final_annotations')
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    print(f"[Info] 正在读取背景: {bg_path}")
    bg = cv2.imread(bg_path)
    if bg is None:
        raise FileNotFoundError(f"无法读取背景图 {bg_path}")
    bg_h, bg_w = bg.shape[:2]

    # --- 关键步骤：生成“两墙之间”的可粘贴掩膜 ---
    print("[Info] 正在生成水体区域掩膜...")
    allowed_mask = get_region_between_walls(bg)

    # 保存掩膜供人工检查 (白色为可粘贴区域)
    debug_mask_path = os.path.join(out_dir, 'debug_water_mask.jpg')
    cv2.imwrite(debug_mask_path, allowed_mask)
    print(f"[Info] 调试掩膜已保存至: {debug_mask_path} (请检查白色区域是否正确)")

    # 获取所有合法坐标点 (y, x)
    allowed_coords = np.column_stack(np.where(allowed_mask > 0))
    if allowed_coords.size == 0:
        raise RuntimeError("在左右池壁之间未找到任何可用空间，请检查背景图或红/蓝阈值。")

    # 重新计算墙体区域（用于后续避免将目标放在墙体上）
    hsv_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    wall_mask = cv2.bitwise_or(cv2.inRange(hsv_bg, lower_red1, upper_red1),
                               cv2.inRange(hsv_bg, lower_red2, upper_red2))
    # 扩展墙体宽度以覆盖厚度
    kernel_w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    wall_thick = cv2.dilate(wall_mask, kernel_w, iterations=3)
    wall_thick = (wall_thick > 0).astype(np.uint8)

    # --- 加载所有素材 ---
    candidates = []
    print("[Info] 正在加载源目标数据...")
    valid_extensions = ('.png', '.jpg', '.jpeg')
    for img_file in os.listdir(source_img_dir):
        if not img_file.lower().endswith(valid_extensions): continue

        base_name = os.path.splitext(img_file)[0]
        xml_path = os.path.join(source_xml_dir, f"{base_name}.xml")
        img_path = os.path.join(source_img_dir, img_file)

        if not os.path.exists(xml_path): continue

        src_img = cv2.imread(img_path)
        if src_img is None: continue

        try:
            _, bboxes, names = parse_xml(xml_path)
            for i, bbox in enumerate(bboxes):
                xmin, ymin, xmax, ymax = bbox
                roi = src_img[ymin:ymax, xmin:xmax]

                # 提取带Alpha通道的目标
                roi_bgra, _ = extract_target_with_alpha(roi)
                if roi_bgra is not None:
                    candidates.append({
                        'img': roi_bgra,
                        'name': names[i]
                    })
        except Exception as e:
            print(f"解析 {xml_path} 出错: {e}")

    if not candidates:
        raise RuntimeError("未找到有效的源目标素材。")
    print(f"[Info] 加载完毕，共有 {len(candidates)} 个可用目标。")

    # --- 开始生成图片 ---
    for img_idx in range(1, num_images + 1):
        composed_img = bg.copy()
        current_annotations = []

        # 确定当前图的目标数量 (默认2-4个)
        tgt_count = num_targets if num_targets is not None else random.randint(1, 3)

        for t_i in range(tgt_count):
            # 随机选一个目标
            candidate = random.choice(candidates)
            tgt_bgra = candidate['img']
            tgt_h, tgt_w = tgt_bgra.shape[:2]
            tgt_alpha = tgt_bgra[:, :, 3]

            placed_success = False
            attempts = 0
            max_attempts = 200  # 每个目标尝试两百次放置

            while attempts < max_attempts:
                attempts += 1

                # 为当前目标计算可放置中心的可行掩码（通过腐蚀保证整个目标区域在 allowed_mask 内）
                # 使用较小的腐蚀核以避免过度收缩（取目标尺寸的一半，且至少为3）
                kh = max(3, min(allowed_mask.shape[0], max(1, int(tgt_h * 0.5))))
                kw = max(3, min(allowed_mask.shape[1], max(1, int(tgt_w * 0.5))))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kh, kw))
                feasible_mask = cv2.erode(allowed_mask, kernel, iterations=1)
                feasible_coords = np.column_stack(np.where(feasible_mask > 0))

                if feasible_coords.shape[0] == 0:
                    # 无可行中心点，放弃此目标
                    break

                rnd_idx = random.randint(0, feasible_coords.shape[0] - 1)
                cy, cx = int(feasible_coords[rnd_idx][0]), int(feasible_coords[rnd_idx][1])

                # 计算左上角
                x0 = cx - tgt_w // 2
                y0 = cy - tgt_h // 2
                x1 = x0 + tgt_w
                y1 = y0 + tgt_h

                # 1. 基础边界检查
                if x0 < 0 or y0 < 0 or x1 > bg_w or y1 > bg_h:
                    continue

                # 2. 精确掩膜匹配检查
                # 截取该区域的 allowed_mask
                mask_crop = allowed_mask[y0:y1, x0:x1]

                # 找出目标中有实体(非透明)的部分
                obj_pixels = tgt_alpha > 10  # bool mask

                if mask_crop.shape != obj_pixels.shape:
                    continue

                # 检查: 目标实体的每一个像素，是否都落在 allowed_mask 的白色区域内?
                check_overlap = mask_crop[obj_pixels]
                if check_overlap.size == 0:
                    continue
                ratio = np.count_nonzero(check_overlap) / check_overlap.size
                if ratio < 0.99:  # 99% 的部分必须在水体掩膜内
                    continue

                # 进一步检查：目标与墙体的重叠比例应很小（例如 <10%），防止目标完全落在墙带上
                wall_crop = wall_thick[y0:y1, x0:x1]
                if wall_crop.shape != obj_pixels.shape:
                    continue
                wall_overlap = np.count_nonzero(wall_crop[obj_pixels])
                wall_ratio = wall_overlap / obj_pixels.sum() if obj_pixels.sum() > 0 else 1.0
                if wall_ratio > 0.1:
                    # 与墙体重叠过多，跳过该位置
                    continue

                # --- 放置成功：使用羽化掩码进行融合（仿 cut_and_enhance 的逻辑） ---
                # 创建单通道 mask（0/255）
                mask_single = (tgt_alpha).astype(np.uint8)
                # 轻度腐蚀 + 羽化，去除边缘伪影
                ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                shrunk = cv2.erode(mask_single, ek, iterations=1)
                feathered = cv2.GaussianBlur(shrunk, (5, 5), 0)

                # 截取背景对应区域的 allowed_mask（之前已命名 allowed_mask）
                sector_crop = allowed_mask[y0:y1, x0:x1]
                if sector_crop.shape != feathered.shape:
                    sector_crop = cv2.resize(sector_crop, (feathered.shape[1], feathered.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)

                # 合并掩码：只有在 sector_crop 有效的位置可以放置
                merged_mask = cv2.bitwise_and((feathered).astype(np.uint8), sector_crop.astype(np.uint8))
                if np.count_nonzero(merged_mask) == 0:
                    # 无重合有效像素，继续尝试其它位置
                    continue

                # 如果合并掩码的有效像素比例极小，跳过
                if np.count_nonzero(merged_mask) < 5:
                    continue

                # 检查与已有目标重叠，避免过多覆盖
                ys_m, xs_m = np.where(merged_mask > 0)
                if ys_m.size == 0:
                    continue
                cand_bbox = [int(x0 + xs_m.min()), int(y0 + ys_m.min()), int(x0 + xs_m.max() + 1),
                             int(y0 + ys_m.max() + 1)]
                cand_area = (cand_bbox[2] - cand_bbox[0]) * (cand_bbox[3] - cand_bbox[1])
                overlap_too_much = False
                for ex in current_annotations:
                    ex_xmin, ex_ymin, ex_xmax, ex_ymax, _ = ex
                    ox1 = max(cand_bbox[0], ex_xmin)
                    oy1 = max(cand_bbox[1], ex_ymin)
                    ox2 = min(cand_bbox[2], ex_xmax)
                    oy2 = min(cand_bbox[3], ex_ymax)
                    if ox1 < ox2 and oy1 < oy2:
                        inter = (ox2 - ox1) * (oy2 - oy1)
                        if inter > 0.3 * cand_area:
                            overlap_too_much = True
                            break
                if overlap_too_much:
                    continue

                # 将 mask 和 feathered 转为三通道
                merged_mask_3ch = cv2.cvtColor(merged_mask, cv2.COLOR_GRAY2BGR) / 255.0

                # 如果目标不是正确尺寸，调整大小
                if tgt_bgra.shape[0] != merged_mask.shape[0] or tgt_bgra.shape[1] != merged_mask.shape[1]:
                    target_resized = cv2.resize(tgt_bgra, (merged_mask.shape[1], merged_mask.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)
                else:
                    target_resized = tgt_bgra

                target_rgb = target_resized[:, :, :3].astype(np.float32)
                roi_area = composed_img[y0:y1, x0:x1].astype(np.float32)
                blended_roi = (target_rgb * merged_mask_3ch + roi_area * (1 - merged_mask_3ch)).astype(np.uint8)
                composed_img[y0:y1, x0:x1] = blended_roi

                # 更新标注：使用 merged_mask 有效像素计算外接矩形
                ys2, xs2 = np.where(merged_mask > 0)
                bxmin = int(x0 + xs2.min())
                bymin = int(y0 + ys2.min())
                bxmax = int(x0 + xs2.max() + 1)
                bymax = int(y0 + ys2.max() + 1)
                current_annotations.append([bxmin, bymin, bxmax, bymax, candidate.get('name', 'target')])

                placed_success = True
                break

            if not placed_success:
                # 仅用于调试，不需要太关注
                pass

                # --- 保存当前图片和XML ---
        save_filename = f"chibi_enhanced_{img_idx:04d}"
        img_save_path = os.path.join(out_images_dir, save_filename + ".png")
        cv2.imwrite(img_save_path, composed_img)

        save_xml(out_ann_dir, save_filename, img_save_path, current_annotations, bg_w, bg_h)
        print(f"[Success] 生成: {save_filename}.png (含 {len(current_annotations)} 个目标)")


def save_xml(save_dir, file_basename, full_path, annotations, w, h):
    """保存Pascal VOC格式XML"""
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = 'chibi'
    ET.SubElement(root, 'filename').text = file_basename + ".png"
    ET.SubElement(root, 'path').text = full_path

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(w)
    ET.SubElement(size, 'height').text = str(h)
    ET.SubElement(size, 'depth').text = '3'

    for ann in annotations:
        xmin, ymin, xmax, ymax, name = ann
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bnd = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bnd, 'xmin').text = str(xmin)
        ET.SubElement(bnd, 'ymin').text = str(ymin)
        ET.SubElement(bnd, 'xmax').text = str(xmax)
        ET.SubElement(bnd, 'ymax').text = str(ymax)

    tree = ET.ElementTree(root)
    tree.write(os.path.join(save_dir, file_basename + ".xml"))


if __name__ == '__main__':
    paste_targets_between_walls(
        bg_path='bg_chibi.jpg',  # 背景图片路径
        source_img_dir='source/images',  # 目标素材图片文件夹
        source_xml_dir='source/annotations',  # 目标素材标注文件夹
        out_dir='complete_enhanced/chibi',  # 结果输出文件夹
        num_targets=None,  # 每张图粘贴的目标数量
        num_images=3000  # 总共生成的图片数量
    )
