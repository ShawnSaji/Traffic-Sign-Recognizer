import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------
# Step 1: Set Your Phone Stream URL
# ---------------------------------------------------------
# Replace with the exact IP shown in your phone app
PHONE_IP = "http://192.168.1.5:8080"  
STREAM_URL = f"{PHONE_IP}/video"      # For IP Webcam app

# ---------------------------------------------------------
# Step 2: Load Models
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO('yolov8n.pt')
YOLO_CLASSES = [0, 1, 2, 9, 11]  # Pedestrians, Bicycles, Cars, Traffic Lights, Stop Signs

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
    print("Running YOLO detection only.")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# Step 3: Connect to Phone Camera Stream
# ---------------------------------------------------------
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Error: Could not open phone stream. Check the IP address and Wi-Fi connection.")
    exit()

print("Streaming live feed from phone. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream disconnected.")
        break

    h, w, _ = frame.shape

    # Part A: Run YOLO Detection
    results = yolo_model(frame, classes=YOLO_CLASSES, conf=0.45, verbose=False)
    annotated_frame = results[0].plot()

    # Part B: Custom Sign Classifier ROI
    if has_classifier:
        box_size = min(h, w) // 2
        top, left = (h - box_size) // 2, (w - box_size) // 2
        
        cv2.rectangle(annotated_frame, (left, top), (left + box_size, top + box_size), (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Place Sign Here", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        roi = frame[top:top+box_size, left:left+box_size]
        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        tensor_img = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = sign_classifier(tensor_img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        label = class_names[predicted_idx.item()]
        score = confidence.item() * 100

        if score > 60:
            cv2.putText(annotated_frame, f"Sign: {label} ({score:.1f}%)", 
                        (left, top + box_size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Phone Camera - Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()