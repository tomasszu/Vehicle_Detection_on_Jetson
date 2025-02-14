import onnxruntime as ort
import numpy as np
import cv2
import torch
import time  # Import the time module

print("CUDA available:", torch.cuda.is_available())

# Load the ONNX model
onnx_model_path = "yolov8n.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])

# Load class labels (COCO dataset)
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
               "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
               "hair drier", "toothbrush"]

# Open video file
video_path = "cam1_cuts.mp4"
cap = cv2.VideoCapture(video_path)


# YOLO expects 640x640 input
input_size = 640
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess_frame(frame):
    """Preprocess frame for YOLOv8 ONNX model"""
    frame_resized = cv2.resize(frame, (input_size, input_size))
    frame_resized = frame_resized / 255.0  # Normalize
    frame_resized = np.transpose(frame_resized, (2, 0, 1))  # Change to (C, H, W)
    frame_resized = np.expand_dims(frame_resized, axis=0).astype(np.float32)  # Add batch dimension
    return frame_resized

#print(outputs.shape)

def post_process(outputs, conf_threshold=0.25, iou_threshold=0.4):
    """Post-process YOLOv8 ONNX model outputs"""
    predictions = np.squeeze(outputs[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return []

    class_ids = np.argmax(predictions[:, 4:], axis=1)
    detected_objects = [(COCO_CLASSES[id], scores[i]) for i, id in enumerate(class_ids)]
    return detected_objects

frame_count = 0
total_inference_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Preprocess frame
    input_tensor = preprocess_frame(frame)

    # Run inference
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_tensor})[0]
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    total_inference_time += inference_time

    # Get detected objects
    detected_objects = post_process(outputs)

    print(f"Frame {frame_count}: Inference time = {inference_time:.2f} ms, Objects: {detected_objects}")

cap.release()
cv2.destroyAllWindows()

avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
print(f"Average inference time per frame: {avg_inference_time:.2f} ms")
