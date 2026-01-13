# 将框绘制到目标上（对不同背景分别绘制每个背景前10张）

import os
import cv2
import xml.etree.ElementTree as ET

# 可视化保存目录
vis_dir = './vis'
os.makedirs(vis_dir, exist_ok=True)

# 背景子目录列表
backgrounds = ['16m', '25m', '50m', '100m']
# 每个背景绘制多少张
num_per_bg = 10

for bg in backgrounds:
    img_dir = os.path.join('./complete_enhanced', bg, 'final_images')
    xml_dir = os.path.join('./complete_enhanced', bg, 'final_annotations')
    vis_subdir = os.path.join(vis_dir, bg)
    os.makedirs(vis_subdir, exist_ok=True)

    for i in range(1, num_per_bg + 1):
        img_path = os.path.join(img_dir, f'final_{i:04}.png')
        xml_path = os.path.join(xml_dir, f'final_{i:04}.xml')
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f'图片读取失败: {img_path}')
            continue
        try:
            tree = ET.parse(xml_path)
        except Exception as e:
            print(f'解析 XML 失败: {xml_path}，{e}')
            continue
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            try:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
            except Exception:
                continue
            # 画框（红色，线宽2）
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            # 保存可视化图片到子目录
            vis_path = os.path.join(vis_subdir, f'vis_{i:04}.png')
            cv2.imwrite(vis_path, img)
