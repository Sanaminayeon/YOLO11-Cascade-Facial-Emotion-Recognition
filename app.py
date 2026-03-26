import streamlit as st
from ultralytics import YOLO
import os
import cv2
import tempfile
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="YOLO11 人像情绪识别系统",
    page_icon="😊",
    layout="wide"
)

st.title("😊 基于 YOLO11 的人像情绪识别系统")
st.sidebar.header("设置")

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model_path = st.sidebar.text_input("模型路径", "best.pt")
try:
    model = load_model(model_path)
    st.sidebar.success("模型加载成功!")
except Exception as e:
    st.sidebar.error(f"模型加载失败: {e}")

conf_threshold = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.25, 0.05)

preprocess_gray = st.sidebar.checkbox(
    "使用灰度预处理 (推荐)", 
    value=True,
    help="FER2013 数据集是灰度图。如果你的模型是用 FER2013 训练的，开启此选项通常能显著提高对彩色图片的识别效果。"
)

fix_bbox = st.sidebar.checkbox(
    "开启人脸定位增强 (Fix Bounding Boxes)",
    value=True,
    help="由于 FER2013 数据集没有背景，训练出的模型倾向于框住整张图。开启此选项将使用 OpenCV 先定位人脸，再用 YOLO 识别情绪，效果更完美。"
)

# 加载 OpenCV 人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 功能选择
app_mode = st.sidebar.selectbox("选择模式", ["主页", "图片检测", "视频检测", "实时摄像头", "训练结果分析"])

if app_mode == "主页":
    st.markdown("""
    ### 欢迎使用人像情绪识别系统
    
    本系统基于最新的 **YOLO11** 模型构建，能够高效地检测人脸并识别其情绪。
    
    #### 支持的功能:
    - **图片检测**: 上传本地图片进行情绪分析。
    - **视频检测**: 上传视频文件进行逐帧分析。
    - **实时摄像头**: 调用本地摄像头进行实时检测。
    - **训练结果分析**: 查看模型训练过程中的各项指标和图表。
    
    #### 情绪类别:
    - Angry (愤怒)
    - Disgust (厌恶)
    - Fear (恐惧)
    - Happy (开心)
    - Sad (悲伤)
    - Surprise (惊讶)
    - Neutral (中性)
    
    请在左侧侧边栏选择模式开始使用。
    """)

elif app_mode == "训练结果分析":
    st.header("模型训练结果分析")
    
    # 默认训练结果路径
    train_dir = "runs/train/emotion_yolo11"
    
    if not os.path.exists(train_dir):
        st.warning(f"未找到训练结果文件夹: `{train_dir}`")
        st.info("请先运行 `python train.py` 进行模型训练。")
    else:
        tab1, tab2, tab3 = st.tabs(["训练指标", "混淆矩阵", "预测样本"])
        
        with tab1:
            st.subheader("训练过程指标 (Results)")
            results_img = os.path.join(train_dir, "results.png")
            if os.path.exists(results_img):
                st.image(results_img, caption="训练过程中的 Loss 和 mAP 指标变化", use_column_width=True)
            else:
                st.info("尚未生成 results.png，可能训练未完成。")
                
        with tab2:
            st.subheader("混淆矩阵 (Confusion Matrix)")
            cm_img = os.path.join(train_dir, "confusion_matrix.png")
            norm_cm_img = os.path.join(train_dir, "confusion_matrix_normalized.png")
            
            if os.path.exists(norm_cm_img):
                st.image(norm_cm_img, caption="归一化混淆矩阵", use_column_width=True)
            elif os.path.exists(cm_img):
                st.image(cm_img, caption="混淆矩阵", use_column_width=True)
            else:
                st.info("尚未生成混淆矩阵。")
                
        with tab3:
            st.subheader("验证集预测样本")
            # 查找 val_batch*_pred.jpg
            pred_imgs = [f for f in os.listdir(train_dir) if f.startswith("val_batch") and f.endswith("_pred.jpg")]
            if pred_imgs:
                selected_pred = st.selectbox("选择批次", pred_imgs)
                st.image(os.path.join(train_dir, selected_pred), caption=f"验证集预测: {selected_pred}", use_column_width=True)
            else:
                st.info("未找到验证集预测示例图。")

elif app_mode == "图片检测":
    st.header("图片情绪识别")
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 读取图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_column_width=True)
        
        if st.button("开始识别"):
            with st.spinner("正在分析中..."):
                display_image = image.copy()
                draw_img = np.array(display_image)
                
                # ------ 模式 A: 两阶段 (OpenCV定位 -> YOLO分类) ------
                if fix_bbox:
                    # 转为 OpenCV 格式
                    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    
                    # 1. 检测人脸
                    faces = face_cascade.detectMultiScale(gray_img, 1.1, 5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        st.success(f"检测到 {len(faces)} 张人脸")
                        for (x, y, w, h) in faces:
                            # 2. 裁剪人脸
                            face_roi = image.crop((x, y, x+w, y+h))
                            
                            # 预处理
                            if preprocess_gray:
                                face_roi = face_roi.convert("L").convert("RGB")
                            
                            # 3. YOLO 识别 (作为分类器使用)
                            results = model.predict(face_roi, verbose=False)
                            
                            # 获取最高置信度的类别
                            if results and results[0].boxes:
                                # 注意：这里虽然通过boxes获取，但我们只关心类别
                                # 或者直接看 probs 如果是分类模型，但这是检测模型，通常会出一个覆盖全图的box
                                cls_id = int(results[0].boxes.cls[0])
                                conf = float(results[0].boxes.conf[0])
                                label = results[0].names[cls_id]
                                
                                # 4. 在原图上绘制
                                cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                text = f"{label} {conf:.2f}"
                                cv2.putText(draw_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        st.image(draw_img, caption="识别结果 (增强模式)", use_column_width=True)
                    else:
                        st.warning("未检测到人脸 (OpenCV)")
                
                # ------ 模式 B: 原生 YOLO 检测 ------
                else:
                    # 预处理: 转为灰度再转回 RGB (模拟训练数据的色彩空间)
                    input_image = image
                    if preprocess_gray:
                        input_image = image.convert("L").convert("RGB")

                    # YOLO 推理
                    results = model.predict(input_image, conf=conf_threshold)
                    
                    # 绘制结果
                    res_plotted = results[0].plot()
                    st.image(res_plotted, caption="识别结果 (原生模式)", use_column_width=True)
                    
                    # 显示检测到的情绪统计
                    st.subheader("检测结果统计:")
                    boxes = results[0].boxes
                    if boxes:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            name = results[0].names[cls_id]
                            st.write(f"- 类别: **{name}**, 置信度: {conf:.2f}")
                    else:
                        st.warning("未检测到人脸/情绪。")

elif app_mode == "视频检测":
    st.header("视频情绪识别")
    uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        stop_button = st.button("停止")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            if not ret:
                break
            
            final_frame = frame.copy()
            
            if fix_bbox:
                # 增强模式
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                for (x, y, w, h) in faces:
                    # 提取 ROI
                    face_roi_bgr = frame[y:y+h, x:x+w]
                    # 预处理
                    if preprocess_gray:
                        roi_gray = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)
                        face_roi_input = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
                    else:
                        face_roi_input = face_roi_bgr
                        
                    # YOLO 推理 (ROI)
                    results = model(face_roi_input, verbose=False, conf=0.1) # 降低阈值，因为是强制分类
                    
                    # 解析结果
                    if results and len(results[0].boxes) > 0:
                        # 找最大置信度的
                        best_box = max(results[0].boxes, key=lambda b: b.conf[0])
                        cls_id = int(best_box.cls[0])
                        conf = float(best_box.conf[0])
                        label = results[0].names[cls_id]
                        
                        # 绘图
                        cv2.rectangle(final_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(final_frame, f"{label} {conf:.2f}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            else:
                # 原生模式
                # 预处理
                process_frame = frame
                if preprocess_gray:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    process_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                # YOLO 推理
                results = model(process_frame, conf=conf_threshold)
                final_frame = results[0].plot()
            
            # 转换颜色空间 BGR -> RGB 用于 Streamlit 显示
            frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")
        
        cap.release()

elif app_mode == "实时摄像头":
    st.header("实时摄像头情绪识别")
    st.info("请确保允许浏览器访问摄像头 (在本地运行此模式通常会弹出新窗口或使用 WebRTC，此处演示使用 cv2 loop，仅在本地 Python 环境有效)")
    
    run = st.checkbox('开启摄像头')
    FRAME_WINDOW = st.image([])
    
    camera = None
    if run:
        camera = cv2.VideoCapture(0)
    
    while run and camera is not None:
        ret, frame = camera.read()
        if not ret:
            st.error("无法读取摄像头")
            break
            
        if not ret:
            st.error("无法读取摄像头")
            break
            
        final_frame = frame.copy()
        
        if fix_bbox:
             # 增强模式
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # 提取 ROI
                face_roi_bgr = frame[y:y+h, x:x+w]
                # 预处理
                if preprocess_gray:
                    roi_gray = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)
                    face_roi_input = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
                else:
                    face_roi_input = face_roi_bgr
                    
                # YOLO 推理 (ROI)
                results = model(face_roi_input, verbose=False, conf=0.1)
                
                # 解析结果
                if results and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: b.conf[0])
                    cls_id = int(best_box.cls[0])
                    conf = float(best_box.conf[0])
                    label = results[0].names[cls_id]
                    
                    # 绘图
                    cv2.rectangle(final_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(final_frame, f"{label} {conf:.2f}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # 预处理 (注意 frame 是 BGR)
            process_frame = frame
            if preprocess_gray:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                process_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
            # 推理
            results = model(process_frame, conf=conf_threshold)
            final_frame = results[0].plot()
        
        # 由于 plot() 返回的是 BGR (OpenCV 格式)，我们需要转为 RGB 给 st.image
        res_plotted_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        
        FRAME_WINDOW.image(res_plotted_rgb)
    
    if camera:
        camera.release()
