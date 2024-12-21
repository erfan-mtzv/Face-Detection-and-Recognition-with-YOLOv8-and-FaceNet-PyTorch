---  

# Face Detection and Recognition with YOLOv8 and FaceNet-PyTorch  

This project implements a robust face detection and recognition pipeline using YOLOv8 for face-keypoint detection and the FaceNet-PyTorch library for face recognition. The system identifies individuals in a video using pre-recorded images in a database. Additionally, it uses a buffer mechanism to improve recognition accuracy by voting across consecutive frames.  

---  

## Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Pipeline Details](#pipeline-details)  
  - [Face Detection with YOLOv8](#face-detection-with-yolov8)  
  - [Face Recognition with FaceNet-PyTorch](#face-recognition-with-facenet-pytorch)  
  - [Buffer Mechanism for Robust Recognition](#buffer-mechanism-for-robust-recognition)  
- [Demo](#demo)  
- [Directory Structure](#directory-structure)  
- [Acknowledgments](#acknowledgments)  

---  

## Overview  
This repository demonstrates a system for detecting and recognizing faces from video footage. The project includes:  
- **Face Detection**: Using YOLOv8 with face-keypoint pretrained weights to detect facial landmarks.  
- **Face Recognition**: Using FaceNet-PyTorch and InceptionResNetV1 with VGGFace2 pretrained weights to extract face embeddings and match identities.  
- **Buffer Mechanism**: Reduces misidentifications by voting across consecutive frames of the same detected face.  

The repository provides a sample dataset and an 8-second CCTV video of two individuals visiting a subway wagon for demonstration purposes.  

---  

## Features  
- Detects faces and tracks their motion using YOLOv8 with face-keypoint pretrained weights.  
- Extracts facial embeddings with FaceNet-PyTorch using VGGFace2 pretrained weights.  
- Recognizes individuals by comparing embeddings against a database using cosine similarity.  
- Implements a buffer mechanism to improve recognition accuracy.  
- Includes a ready-to-use notebook for step-by-step execution.  

---  

## Requirements  
- **Python**: 3.8+  
- **Libraries**:  
  - PyTorch  
  - FaceNet-PyTorch  
  - Ultralytics (YOLOv8)  
  - OpenCV  
  - NumPy  
- **Hardware**: A GPU is recommended for faster processing.  

---  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone[ https://github.com/erfan-mtzv/Face-Detection-and-Recognition-with-YOLOv8-and-FaceNet-PyTorch.git
   cd Face-Detection-and-Recognition-with-YOLOv8-and-FaceNet-PyTorch
   ```  
2. Install the required libraries

3. Clone the FaceNet-PyTorch repository:  
   ```bash  
   git clone https://github.com/timesler/facenet-pytorch.git  
   ```  

---  

## Usage  
1. Open the Jupyter Notebook provided in the repository.  
2. Follow the notebook's steps to:  
   - Detect faces in the video using YOLOv8.  
   - Extract embeddings for the detected faces.  
   - Recognize identities from the video using the database.  
   - Compare results with and without the buffer mechanism.  
3. View the results, including:  
   - The **video result without buffer** mode.  
   - The **video result with buffer** mode.  

---  

## Pipeline Details  

### Face Detection with YOLOv8  
- Utilizes YOLOv8 with face-keypoint pretrained weights.  
- Detects five keypoints per face: both eyes, nose, and two edges of the lips.  
- Tracks faces across video frames.  

### Face Recognition with FaceNet-PyTorch  
- Uses InceptionResNetV1 with VGGFace2 pretrained weights.  
- Extracts 512-dimensional embedding vectors for detected faces.  
- Matches embeddings with a prebuilt database using cosine similarity:  

  **Cosine Similarity Formula**:  

$\text{Similarity} = \frac{\text{Embedding}_1 \cdot \text{Embedding}_2}{\|\text{Embedding}_1\| \|\text{Embedding}_2\|}$


### Buffer Mechanism for Robust Recognition  
- Buffers consecutive frames of a detected face ID.  
- Votes across `n` frames to assign the final identity.  
- Reduces misidentifications and improves accuracy.  

---  

## Demo  
The repository includes:  
- A **video**: CCTV footage of two individuals (8 seconds).  
- A **database**: Contains images of Kathy Hochul (4 images) and Janno Lieber (2 images).  
- The Jupyter Notebook with step-by-step instructions.  

### Demo Videos  

#### Without Buffer  
![](https://github.com/erfan-mtzv/Face-Detection-and-Recognition-with-YOLOv8-and-FaceNet-PyTorch/blob/main/result-videos/without-buffer-result.gif)

#### With Buffer (buffer_size=5)
![](https://github.com/erfan-mtzv/Face-Detection-and-Recognition-with-YOLOv8-and-FaceNet-PyTorch/blob/main/result-videos/buffer-result.gif)


---  

## Directory Structure  
```  
📁 face_detection_recognition/  
├── 📂 database/               # Database with images of known individuals  
├── 📂 result-videos/          # Results of the pipeline  
├── 📄 face_detection_recognition.ipynb # Jupyter Notebook  
├── 📄 subway.mp4              # A sample input 8-second video  
└── 📜 README.md               # Project README  
```  

---  

## Acknowledgments  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for face detection and tracking.  
- [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) for face recognition.  
- Inspiration from real-world use cases in video surveillance and identity management systems.  

---  
