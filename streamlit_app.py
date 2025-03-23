import streamlit as st
import cv2
import torch
import time
import av
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Streamlit UI Configuration
st.set_page_config(page_title="Gods' Eye Object Detection", layout="wide")

st.markdown("<h1 style='text-align: center; color: #ff66b2;'>Ultralytics YOLO Streamlit Application</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: blue;'>Experience real-time security detection with the power of Ultralytics YOLO! 🚀</h3>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Video")
video_source = st.sidebar.selectbox("Select Input Source", ["Webcam", "Upload File", "RTMP Stream"])

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

if video_source == "Webcam":
    # Ensure session state exists
    if "webcam_running" not in st.session_state:
        st.session_state["webcam_running"] = False

    # FPS tracking
    prev_time = time.time()
    fps_latency_area = st.sidebar.empty()
    # Video Processing Callback
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        global prev_time

        start_time = time.time()
        img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format

        # Run YOLO Inference
        results = model(img, conf=confidence_threshold, iou=iou_threshold)

        for r in results:
            annotated_frame = r.plot()  # Draw bounding boxes

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        # st.sidebar.markdown(f"**FPS: {fps:.2f}**")
        print(f"FPS: {fps:.2f}")

        # Latency Calculation
        latency = (time.time() - start_time) * 1000  # Convert to ms
        print(f"Latency per frame: {latency:.2f} ms")
        fps_latency_area.markdown(f"**FPS: {fps:.2f} | Latency: {latency:.2f} ms | Frames Processed: {frame_count}**")

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # UI for Start/Stop Buttons
    st.title("🔍 Real-time YOLO Object Detection with WebRTC")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Start"):
        st.session_state["webcam_running"] = True

    if col2.button("Stop"):
        st.session_state["webcam_running"] = False

    # Start WebRTC only when "Start" is clicked
    if st.session_state["webcam_running"]:
        webrtc_streamer(
            key="yolo-stream",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},  # Video-only
        )
    else:
        st.warning("Click 'Start' to begin webcam detection.")

elif video_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["png", "jpg", "mp4", "avi", "mov"])
    start = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")
    
    if uploaded_file is not None:
        st.sidebar.success("File Uploaded Successfully")

        if uploaded_file.type.startswith("image"):
            stframe = st.empty()
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            fps_latency_area = st.sidebar.empty()
            # Run YOLO inference on the image
            if start:
                image_array = np.array(image)  # Convert to NumPy array
                start_time = time.time()
                results = model(image_array, conf=confidence_threshold, iou=iou_threshold)

                for r in results:
                    annotated_frame = r.plot()
                
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                current_time = time.time()
                latency = (time.time() - start_time) * 1000 
                fps_latency_area.markdown(f"**Latency: {latency:.2f} ms**")

        else:
            st.video(uploaded_file)
            
            if uploaded_file is not None:
                if "video" not in st.session_state:
                    st.session_state.streaming = False
                    
                if start:
                    st.session_state.streaming = True  # Start streaming

                if stop:
                    st.session_state.streaming = False
                
                if st.session_state.streaming:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                        tmpfile.write(uploaded_file.read())
                        temp_file_path = tmpfile.name
                        
                    st.sidebar.success("File Uploaded Successfully")
                    
                    cap = cv2.VideoCapture(temp_file_path)
                    
                    if not cap.isOpened():
                        st.error("Error opening cap file")
                    else:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        output_path = tempfile.mktemp(suffix='.mp4')
                        out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
                        
                        frame_count = 0
                        prev_time = time.time()
                        
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        processed_frames = 0
                        
                        fps_latency_area = st.sidebar.empty()
                        
                        progress_bar = st.progress(0)
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: 
                                break
                            
                            start_time = time.time()
                            
                            results = model(frame, conf=confidence_threshold, iou=iou_threshold)

                            for r in results:
                                annotated_frame = r.plot()
                            
                            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                            out.write(frame_rgb)

                            current_time = time.time()
                            fps = 1 / (current_time - prev_time)
                            prev_time = current_time
                            frame_count += 1

                            # Log FPS and latency
                            latency = (time.time() - start_time) * 1000  # Convert to ms
                            fps_latency_area.markdown(f"**FPS: {fps:.2f} | Latency: {latency:.2f} ms | Frames Processed: {frame_count}**")
                            processed_frames += 1
                            progress_bar.progress(processed_frames / total_frames)

                            
                        cap.release()
                        out.release()
                
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Processed Video",
                            data=f,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
            
                    os.remove(temp_file_path)
            
elif video_source == "RTMP Stream":
    rtmp_url = st.sidebar.text_input("Enter RTMP URL", value="")
    start = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")

    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    if start:
        st.session_state.streaming = True  # Start streaming

    if stop:
        st.session_state.streaming = False  # Stop streaming

    if st.session_state.streaming and rtmp_url:
        cap = cv2.VideoCapture(rtmp_url)

        if not cap.isOpened():
            st.error("Error opening RTMP stream. Check the URL.")
        else:
            frame_count = 0
            prev_time = time.time()
            fps_latency_area = st.sidebar.empty()
            stframe = st.empty()

            while st.session_state.streaming:
                ret, frame = cap.read()
                if not ret:
                    st.error("Stream ended or could not be read.")
                    break

                start_time = time.time()

                # Run YOLO model
                results = model(frame, conf=confidence_threshold, iou=iou_threshold)

                for r in results:
                    annotated_frame = r.plot()
                        
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # FPS Calculation
                current_time = time.time()
                elapsed_time = current_time - prev_time
                fps = 1 / elapsed_time if elapsed_time > 0 else 0
                prev_time = current_time
                frame_count += 1

                # Display FPS and Latency
                latency = (time.time() - start_time) * 1000  # Convert to ms
                fps_latency_area.markdown(f"**FPS: {fps:.2f} | Latency: {latency:.2f} ms | Frames Processed: {frame_count}**")

                # Show frame
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)

                # Stop check
                if not st.session_state.streaming:
                    break  # Exit loop when Stop is clicked

            cap.release()
            st.success("Streaming stopped successfully.")
        
                    
        
    

st.success("Model loaded successfully!")