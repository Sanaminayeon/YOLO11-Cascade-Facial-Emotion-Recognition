# 基于 YOLO11 的人像情绪识别系统

这是一个基于 YOLO11 模型的人像情绪识别系统。包含模型训练、推理脚本以及一个可视化的 Web 界面。

## 目录结构
- `train.py`: 训练模型的脚本
- `predict.py`: 使用摄像头或视频文件进行推理的脚本
- `app.py`: Streamlit 可视化 Web 应用界面
- `data.yaml`: 数据集配置文件
- `requirements.txt`: 项目依赖库
- `ultralytics-main/`: YOLO 源码（如果需要修改底层代码）

## 环境安装

1. 建议使用 Python 3.8 或更高版本。
2. 安装依赖库：
   ```bash
   pip install -r requirements.txt
   ```
   或者安装本地的 ultralytics（如果有修改）：
   ```bash
   cd ultralytics-main
   pip install -e .
   cd ..
   ```

## 数据集准备

在训练之前，你需要准备好 YOLO 格式的情绪数据集。
1. 在项目根目录下创建 `datasets` 文件夹。
2. 将数据按照以下结构组织：
   ```
   datasets/
   └── emotion/
       ├── images/
       │   ├── train/
       │   └── val/
       └── labels/
           ├── train/
           └── val/
   ```
3. 确保 label 文件 (.txt) 中的类别 ID 与 `data.yaml` 中的 `names` 对应：
   - 0: angry
   - 1: disgust
   - 2: fear
   - 3: happy
   - 4: sad
   - 5: surprise
   - 6: neutral

## 模型训练

数据集准备好后，运行以下命令开始训练：

```bash
python train.py
```

训练过程中的权重文件保存在 `runs/train/emotion_yolo11/weights/` 目录下。

## 运行系统

### 1. 简单推理 (命令行/窗口)
使用默认摄像头进行实时检测：
```bash
python predict.py
```

### 2. 可视化 Web 界面 (推荐)
启动 Streamlit 应用：
```bash
streamlit run app.py
```
启动后，浏览器会自动打开，你可以在界面上选择：
- **图片检测**: 上传图片进行识别
- **视频检测**: 上传视频进行识别
- **实时摄像头**: 直接调用摄像头

## 注意事项

- 首次运行会自动下载 `yolo11n.pt` 预训练模型。
- 如果训练效果不佳，可以尝试增加 `epochs` 或使用更大的模型（如 `yolo11s.pt`, `yolo11m.pt`）。
- 请确保电脑连接了摄像头以使用实时检测功能。
