import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
TARGET_CLASSES = {"person", "car", "motorcycle", "truck", "bus"}

model = YOLO(MODEL_PATH)

def detect_objects(frame):
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
    detections = []
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if label in TARGET_CLASSES:
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'box': box.xyxy[0].tolist()  # bounding box
                })
    return detections