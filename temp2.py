import streamlit as st
import cv2
import torch
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Streamlit UI Configuration
st.set_page_config(page_title="Gods' Eye Object Detection", layout="wide")

st.markdown("<h1 style='text-align: center; color: #ff66b2;'>Ultralytics YOLO Streamlit Application</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: blue;'>Experience real-time security detection with the power of Ultralytics YOLO! 🚀</h3>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Video")
video_source = st.sidebar.selectbox("Select Input Source", ("webcam", "Upload File"))

st.sidebar.title("Enable Tracking")
enable_tracking = st.sidebar.radio("Enable Tracking", ["Yes", "No"])

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

st.sidebar.title("Model")
model_path = st.sidebar.selectbox("Select Model", ["best.pt", "yolov8n.pt"])

# Load YOLO model
@st.cache_resource  # Cache model loading
def load_model():
    return YOLO(model_path)

model = load_model()

# Webcam or File Processing
if video_source == "webcam":
    st.sidebar.title("Webcam Detection")
    
    start = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")
    cap = cv2.VideoCapture(0)
    
    if start:
        stframe = st.empty()
        fps_text = st.empty()

        # Store session state to manage the webcam state
        st.session_state["webcam_running"] = True
        prev_time = time.time()

    if stop:
        st.session_state["webcam_running"] = False

    if "webcam_running" in st.session_state and st.session_state["webcam_running"]:
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame")
                break
            
            # Start for FPS
            current_time = time.time()
            
            # Run YOLO inference
            results = model(frame, conf=confidence_threshold, iou=iou_threshold)

            for r in results:
                annotated_frame = r.plot()

            # Convert frame to RGB (for Streamlit display)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            st.session_state["fps"] = fps
            
            # Display the annotated frame
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
            
            end_time = time.time()  # End time for latency calculation
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"Latency per frame: {latency:.2f} ms")
            # Check if Stop button was pressed
            if not st.session_state["webcam_running"]:
                cap.release()
                break

elif video_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["png", "jpg", "mp4", "avi", "mov"])
    start = st.sidebar.button("Start")
    
    if uploaded_file is not None:
        st.sidebar.success("File Uploaded Successfully")
        if uploaded_file.type.startswith("image"):
            stframe = st.empty()
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width="True")
            # detect image
            if start: 
                image_array = np.array(image) #Convert to Numpy Array
                results = model(image_array, conf=confidence_threshold, iou=iou_threshold)
                
                for r in results:
                    annotated_frame = r.plot()

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                
        else:
            st.video(uploaded_file)
            stframe = st.empty()

st.success("Model loaded successfully!")
