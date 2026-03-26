from ultralytics import YOLO
import os

def train_model():
    # Load a model
    # 使用 yolo11n.pt 作为预训练模型，如果想要更高精度可以使用 yolo11s.pt, yolo11m.pt 等
    model = YOLO("yolo11n.pt")  

    # Train the model
    # data: 数据集配置文件路径 (data.yaml)
    # epochs: 训练轮数
    # imgsz: 图片大小
    # project: 保存路径的项目名称
    # name: 保存结果的文件夹名称
    results = model.train(
        data="data.yaml", 
        epochs=100, 
        imgsz=640, 
        device="0", # Set to "cpu" if no GPU
        project="runs/train", 
        name="emotion_yolo11"
    )
    
    # Evaluate performance
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    
    # Export the model
    model.export(format="onnx")

if __name__ == '__main__':
    # 确保 data.yaml 存在
    if not os.path.exists("data.yaml"):
        print("错误: 找不到 data.yaml 文件。请确保已准备好数据集并配置了 data.yaml。")
    else:
        train_model()
