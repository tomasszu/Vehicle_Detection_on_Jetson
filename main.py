import onnxruntime as ort
import numpy as np
import cv2
import torch
import time  # Import the time module

print("CUDA available:")
print(torch.cuda.is_available())

# Load the ONNX model
onnx_model_path = "yolov8n.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

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

# Read and preprocess an image
image_path = "cars.jpg"  # Change this to your image
image = cv2.imread(image_path)
h, w, _ = image.shape


# YOLO expects 640x640 input
input_size = 640
image_resized = cv2.resize(image, (input_size, input_size))
image_resized = image_resized / 255.0  # Normalize
image_resized = np.transpose(image_resized, (2, 0, 1))  # Change to (C, H, W)
image_resized = np.expand_dims(image_resized, axis=0).astype(np.float32)  # Add batch dimension

start_time = time.time()  # Start the timer

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], {input_name: image_resized})[0]

inference_time = (time.time() - start_time) * 1000  # Convert time to milliseconds
print(outputs.shape)

def post_process(outputs, conf_threshold=0.25, iou_threshold=0.4):
    """
    Filters YOLOv8 ONNX model outputs using confidence threshold and NMS.
    Returns a list of detected objects with class names and confidence scor

    """
    predictions = np.squeeze(outputs[0]).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    #print(scores)


    if len(scores) == 0:
        return [], [], []
    
    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    #print(class_ids)
    

    detected_objects = []
    for i, id in enumerate(class_ids):
        detected_objects.append((COCO_CLASSES[id], scores[i]))
    return detected_objects

# Get detected objects
detected_objects = post_process(outputs)

print(detected_objects)
print(f"Inference time: {inference_time:.2f} ms")