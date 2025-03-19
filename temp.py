if "webcam_running" not in st.session_state:
    st.session_state["webcam_running"] = False

# Define WebRTC VideoTransformer
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_time = time.time()  # For FPS Calculation

    def transform(self, frame):
        start_time = time.time()

        # Convert frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO Inference
        results = model(img, conf=confidence_threshold, iou=iou_threshold)

        for r in results:
            annotated_frame = r.plot()  # Draw bounding boxes

        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        st.sidebar.markdown(f"**FPS: {fps:.2f}**")

        # Latency Calculation
        latency = (time.time() - start_time) * 1000  # Convert to ms
        print(f"Latency per frame: {latency:.2f} ms")

        return frame_rgb

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
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},  # Video-only
    )
else:
    st.warning("Click 'Start' to begin webcam detection.")
