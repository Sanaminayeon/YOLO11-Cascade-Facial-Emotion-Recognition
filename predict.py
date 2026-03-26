import cv2
from ultralytics import YOLO
import time

def run_inference(source=0):
    """
    运行推理
    source: 0 for webcam, or path to image/video file
    """
    # Load the trained model
    # 注意：这里默认加载预训练的 yolo11n.pt，实际使用时请替换为你训练好的模型路径
    # 例如: model = YOLO("runs/train/emotion_yolo11/weights/best.pt")
    model = YOLO("yolo11n.pt") 

    # Open the video source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("YOLO11 Emotion Recognition", annotated_frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 默认使用摄像头
    run_inference(0)
