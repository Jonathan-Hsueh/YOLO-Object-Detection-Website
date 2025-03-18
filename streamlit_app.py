from ultralytics import solutions

inf = solutions.Inference(
    model="best.pt",  # you can use any model that Ultralytics support, i.e. YOLO11, YOLOv10
)

inf.inference()  # this will run the inference on the image