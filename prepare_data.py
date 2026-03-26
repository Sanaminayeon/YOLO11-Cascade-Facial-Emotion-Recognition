import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置部分 =================
# 原始 FER2013 数据集路径 (请修改为你下载解压后的实际路径)
# 假设结构为:
# raw_dataset/
#   train/
#     angry/
#     disgust/
#     ...
#   test/ (或 validation)
#     angry/
#     ...
RAW_DATASET_DIR = "datasets/raw_fer2013"

# YOLO 格式输出路径
OUTPUT_DIR = "datasets/emotion"

# 情绪标签映射 (FER2013 文件夹名 -> YOLO class ID)
# 必须与 data.yaml 中的 names 顺序一致
CLASS_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

# ===========================================

def convert_to_yolo_detection(source_dir, output_dir, split="train"):
    """
    将分类数据集转换为 YOLO 检测格式 (Bounding Box = 全图)
    """
    print(f"正在转换 {split} 集...")
    
    # 目标目录
    images_dir = os.path.join(output_dir, "images", split)
    labels_dir = os.path.join(output_dir, "labels", split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 遍历每个类别文件夹
    for emotion_name, class_id in CLASS_MAP.items():
        emotion_dir = os.path.join(source_dir, split, emotion_name)
        
        # 兼容 folder 命名差异 (例如 val vs test)
        if not os.path.exists(emotion_dir) and split == "val":
             alt_dir = os.path.join(source_dir, "test", emotion_name)
             if os.path.exists(alt_dir):
                 emotion_dir = alt_dir
        
        if not os.path.exists(emotion_dir):
            print(f"警告: 找不到文件夹 {emotion_dir}，跳过该类别。")
            continue

        # 获取该类别下所有图片
        files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for file_name in tqdm(files, desc=f"Processing {emotion_name}"):
            src_path = os.path.join(emotion_dir, file_name)
            
            # 1. 复制/处理图片
            # 这里我们直接复制，若需要转格式可以用 cv2 读取再保存
            dst_img_name = f"{emotion_name}_{file_name}"
            dst_img_path = os.path.join(images_dir, dst_img_name)
            shutil.copy(src_path, dst_img_path)

            # 2. 生成 Label 文件
            # FER2013 都是裁剪好的人脸，所以 Bounding Box 设为整个图片
            # YOLO 格式: class_id x_center y_center width height (归一化 0-1)
            # 全图覆盖即: class_id 0.5 0.5 1.0 1.0
            label_content = f"{class_id} 0.5 0.5 1.0 1.0"
            
            dst_label_name = os.path.splitext(dst_img_name)[0] + ".txt"
            dst_label_path = os.path.join(labels_dir, dst_label_name)
            
            with open(dst_label_path, "w") as f:
                f.write(label_content)

if __name__ == "__main__":
    if not os.path.exists(RAW_DATASET_DIR):
        print(f"错误: 原始数据集路径 '{RAW_DATASET_DIR}' 不存在。")
        print("请下载 FER2013 数据集并解压，然后修改脚本中的 RAW_DATASET_DIR 变量。")
    else:
        # 转换训练集
        convert_to_yolo_detection(RAW_DATASET_DIR, OUTPUT_DIR, "train")
        # 转换验证集 (FER2013 通常叫 test)
        convert_to_yolo_detection(RAW_DATASET_DIR, OUTPUT_DIR, "val")
        
        print("\n转换完成！")
        print(f"数据已保存至: {OUTPUT_DIR}")
        print("现在你可以运行 python train.py 进行训练了。")
