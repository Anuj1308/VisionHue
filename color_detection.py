import cv2
import numpy as np

# Load YOLO network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

color_ranges = {
    "Red": ([0, 120, 70], [10, 255, 255]),
    "Green": ([36, 100, 100], [86, 255, 255]),
    "Blue": ([94, 80, 2], [126, 255, 255]),
    "Yellow": ([15, 100, 100], [35, 255, 255]),
    "White": ([0, 0, 200], [180, 30, 255]),
    "Black": ([0, 0, 0], [180, 255, 30])
}

cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # If object confidence is high
                center_x, center_y, w, h = (obj[:4] * [width, height, width, height]).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                # Crop object region
                cropped_obj = frame[y:y+h, x:x+w]
                if cropped_obj.shape[0] > 0 and cropped_obj.shape[1] > 0:
                    # Convert to HSV
                    hsv = cv2.cvtColor(cropped_obj, cv2.COLOR_BGR2HSV)
                    detected_color = "Unknown"

                    for color, (lower, upper) in color_ranges.items():
                        lower = np.array(lower, dtype=np.uint8)
                        upper = np.array(upper, dtype=np.uint8)

                        mask = cv2.inRange(hsv, lower, upper)
                        if cv2.countNonZero(mask) > 500:
                            detected_color = color
                            break

                    # Draw object bounding box and detected color
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{classes[class_id]} - {detected_color}", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Object & Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
