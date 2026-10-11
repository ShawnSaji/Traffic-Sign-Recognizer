import cv2

print("Scanning for Iriun Camera Index...")

for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f" SUCCESS: Active Camera Found at INDEX {idx}")
            cap.release()
            break
        cap.release()
    else:
        print(f" Index {idx} empty.")