# Object & Color Detection using YOLO and OpenCV

## Project Overview
This project implements **YOLO (You Only Look Once) v4** with **OpenCV** to perform real-time object detection and color identification using a webcam. It utilizes **deep learning-based object detection** and **HSV-based color recognition** to classify objects and determine their dominant color. 

## Features
- **Real-time Object Detection** using YOLOv4.
- **Color Identification** using HSV color space.
- **Webcam Integration** for live detection.
- **Bounding Boxes & Labels** for detected objects.
- **Customizable Color Definitions** via `colors.json`.

## Technologies Used
| Technology  | Purpose |
|-------------|---------|
| **Python**  | Main programming language |
| **OpenCV**  | Image processing and computer vision |
| **YOLOv4**  | Deep learning-based object detection |
| **NumPy**   | Array operations and mathematical computations |
| **COCO Dataset** | Pre-trained dataset for object detection |

---

## How YOLO Works?
**YOLO (You Only Look Once)** is a **fast and efficient** object detection algorithm that detects objects in real-time by:
1. **Dividing an image into a grid** and assigning each grid cell a responsibility for detecting an object.
2. **Applying a single convolutional neural network (CNN)** to predict bounding boxes and class probabilities.
3. **Using Non-Maximum Suppression (NMS)** to filter overlapping detections and keep the most confident prediction.

### Why YOLO?
- **Speed**: Processes an entire image in a single pass through the neural network.
- **Accuracy**: Outperforms older region-based CNNs (R-CNN, Faster R-CNN) in detection tasks.
- **Efficiency**: Suitable for real-time applications such as **autonomous driving, surveillance, and robotics**.

---

## About OpenCV
**OpenCV (Open Source Computer Vision Library)** is an open-source library that provides real-time image processing and computer vision capabilities.

### Advantages of OpenCV:
- **Highly optimized** for image processing.
- **Multi-platform** (Windows, Linux, Mac, and mobile OS).
- **Supports deep learning frameworks** like TensorFlow and PyTorch.
- **Includes built-in functions** for face detection, motion tracking, and object recognition.

---

## Dataset Used: COCO (Common Objects in Context)
YOLOv4 is trained on the **COCO dataset**, which contains:
- **80 object classes** (e.g., car, person, dog, bicycle, etc.).
- **Over 330,000 labeled images**.
- **Context-rich scenes** to improve object recognition accuracy.

For custom training, the dataset can be modified to detect specific objects.

---

## Detection Methodology
1. **YOLO Model Initialization**:  
   - Loads **YOLOv4 weights** and **configuration file**.
   - Loads the **coco.names** file containing class labels.

2. **Preprocessing Input**:  
   - Captures frames from the webcam.  
   - Resizes images for YOLO model compatibility.  

3. **Object Detection**:  
   - YOLO processes the frame, predicting **bounding boxes** and **confidence scores**.  
   - Applies **Non-Maximum Suppression (NMS)** to eliminate redundant detections.  

4. **Color Detection**:  
   - Extracts the **dominant color** within the bounding box using HSV color space.  
   - Maps colors to predefined labels from `colors.json`.  

5. **Result Display**:  
   - Draws bounding boxes around detected objects.  
   - Displays **object labels and detected color** on the frame.  

---

## Project Structure
```bash
/colour-detection-yolo-opencv
├── yolov4.weights          # YOLO pre-trained weights
├── yolov4.cfg              # YOLO configuration file
├── coco.names              # YOLO class labels
├── color_detection.py      # Main script for detection
├── README.md               # Project documentation
