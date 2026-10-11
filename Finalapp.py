import os
# Force OpenCV to ignore network protocols (stops FFmpeg TCP socket errors)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------
# Step 1: Connect to Phone Camera via Iriun USB
# ---------------------------------------------------------
IRIUN_INDEX = 1  # Usually 1 or 2 for Iriun Virtual Camera

cap = cv2.VideoCapture(IRIUN_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency for USB

if not cap.isOpened():
    print(f"Index {IRIUN_INDEX} failed. Trying Index 2...")
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Could not find USB camera. Ensure Iriun Desktop is running and connected via USB.")
    exit()

print("USB Camera Connected Successfully!")

# ---------------------------------------------------------
# Step 2: Load Models (Fixed Dynamic Class Loading)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

# 1. Load YOLOv8 (Detects Pedestrians, Cars, Traffic Lights, Stop Signs)
yolo_model = YOLO('yolov8n.pt')
YOLO_CLASSES = [0, 1, 2, 9, 11]

# 2. Load Custom ResNet Sign Classifier
classifier_path = 'traffic_sign_model_2.pth'

try:
    checkpoint = torch.load(classifier_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Extract actual output class count directly from the saved weight tensor shape
    num_classes = state_dict['fc.weight'].shape[0]
    
    class_names = checkpoint.get('class_names', [f"Class_{i}" for i in range(num_classes)])
    
    # Rebuild ResNet18 structure with exact saved class dimensions
    sign_classifier = models.resnet18()
    sign_classifier.fc = nn.Linear(sign_classifier.fc.in_features, num_classes)
    sign_classifier.load_state_dict(state_dict)
    sign_classifier = sign_classifier.to(device)
    sign_classifier.eval()
    has_classifier = True
    print(f"Success: '{classifier_path}' loaded ({num_classes} classes configured).")
except FileNotFoundError:
    has_classifier = False
    print(f"Error: '{classifier_path}' not found in the project folder. Running YOLO only.")
except Exception as e:
    has_classifier = False
    print(f"Error loading classifier: {e}")

# Image preprocessing for the ResNet model
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# Step 3: Stream Loop
# ---------------------------------------------------------
print("Starting live detection feed. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera stream disconnected.")
        break

    h, w, _ = frame.shape

    # --- Part A: YOLO Object Detection ---
    results = yolo_model(frame, classes=YOLO_CLASSES, conf=0.45, verbose=False)
    annotated_frame = results[0].plot()

    # --- Part B: Custom Sign Classifier (Center Target Box) ---
    if has_classifier:
        box_size = min(h, w) // 2
        top, left = (h - box_size) // 2, (w - box_size) // 2
        
        # Crop region from clean raw frame
        roi = frame[top:top+box_size, left:left+box_size]
        
        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        tensor_img = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = sign_classifier(tensor_img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        idx = predicted_idx.item()
        label = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        score = confidence.item() * 100

        # Draw targeting box
        cv2.rectangle(annotated_frame, (left, top), (left + box_size, top + box_size), (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Place Sign Here", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if score > 60:
            cv2.putText(annotated_frame, f"Sign: {label} ({score:.1f}%)", 
                        (left, top + box_size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display video output
    cv2.imshow("Phone Feed - Pedestrian & Indian Traffic Sign Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()