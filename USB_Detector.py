import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------
# Step 1: Select Iriun USB Camera Index
# ---------------------------------------------------------
# Change IRIUN_INDEX to 1 or 2 (whichever showed your phone feed above)
IRIUN_INDEX = 1  

cap = cv2.VideoCapture(IRIUN_INDEX, cv2.CAP_DSHOW)

# Reduce buffer size to 1 frame for zero latency over USB
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"Could not open camera at index {IRIUN_INDEX}. Trying index 2...")
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not hook into Iriun phone camera stream.")
    exit()

print("Successfully connected to Phone Camera via Iriun USB!")

# ---------------------------------------------------------
# Step 2: Load Models
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv8 for Pedestrians, Cars, Traffic Lights, Stop Signs
yolo_model = YOLO('yolov8n.pt')
YOLO_CLASSES = [0, 1, 2, 9, 11]

# Load Custom ResNet Sign Classifier
classifier_path = 'traffic_sign_model.pth'
try:
    checkpoint = torch.load(classifier_path, map_location=device)
    class_names = checkpoint['class_names']
    
    sign_classifier = models.resnet18()
    sign_classifier.fc = nn.Linear(sign_classifier.fc.in_features, len(class_names))
    sign_classifier.load_state_dict(checkpoint['model_state_dict'])
    sign_classifier = sign_classifier.to(device)
    sign_classifier.eval()
    has_classifier = True
    print("Sign classifier loaded successfully.")
except FileNotFoundError:
    has_classifier = False
    print(f"Warning: '{classifier_path}' not found. Running YOLO detection only.")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# Step 3: Stream & Inference Loop
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Phone camera stream disconnected.")
        break

    h, w, _ = frame.shape

    # Part A: Run YOLO Object Detection
    results = yolo_model(frame, classes=YOLO_CLASSES, conf=0.45, verbose=False)
    annotated_frame = results[0].plot()

    # Part B: Custom Sign Classifier Target Area
    if has_classifier:
        box_size = min(h, w) // 2
        top, left = (h - box_size) // 2, (w - box_size) // 2
        
        # Crop region from unannotated raw frame
        roi = frame[top:top+box_size, left:left+box_size]
        
        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        tensor_img = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = sign_classifier(tensor_img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        label = class_names[predicted_idx.item()]
        score = confidence.item() * 100

        # Overlay green target box and predicted sign label
        cv2.rectangle(annotated_frame, (left, top), (left + box_size, top + box_size), (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Place Sign Here", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if score > 60:
            cv2.putText(annotated_frame, f"Sign: {label} ({score:.1f}%)", 
                        (left, top + box_size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Phone Feed - Pedestrian & Traffic Sign Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()