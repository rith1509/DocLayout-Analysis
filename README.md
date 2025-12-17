# 📄 Document Layout Analysis Pipeline
> **Rotation-Aware YOLOv10 + OCR + Multilingual Orientation Detection**

This repository implements a robust document layout analysis system designed to handle real-world challenges such as arbitrary rotations, perspective distortion, and multilingual scripts.

---

## 📌 Overview

This pipeline combines state-of-the-art object detection (**YOLOv10**) with advanced preprocessing and **OCR (Tesseract)** to extract structural intelligence from scanned or photographed documents. 

### Key Capabilities:
* **Rotation Robustness:** Automatic detection and correction of 0°, 90°, 180°, and 270° orientations.
* **Perspective Correction:** Four-point transform to "flatten" photographed documents.
* **Multilingual Support:** OSD (Orientation and Script Detection) for Latin, Arabic, Indic, and Sinhalese scripts.
* **Structural Analysis:** High-precision detection of `text`, `titles`, `images`, `tables`, and `figures`.


## ✨ Features

* **Rotation-Aware Detection:** Uses metadata-driven deskewing or Tesseract OSD to normalize images before inference.
* **Coordinate Mapping:** Automatically maps predicted bounding boxes from the corrected image back to the original (rotated/distorted) source coordinates.
* **OCR Integration:** Extracts semantic text data and performs line-grouping/paragraph detection.
* **Automated Pipeline:** Full script for dataset splitting, YOLO-format conversion, and COCO-style submission generation.

---

<img width="612![Uploading doc_00378.png…]()
" height="792" alt="doc_02749" src="https://github.com/user-attachments/assets/7935e7ad-7f29-4fe6-b38d-1b94ab1433af" />
<img width="612" height="792" alt="doc_00098" src="https://github.com/user-attachments/assets/652897f4-02db-4a02-b6d8-3c65b10fd771" />


## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `train.py` | Main pipeline for dataset preparation, YOLOv10 training, and submission export. |
| `detect_multilingual.py` | Advanced inference script handling multilingual OSD and fine-skew correction. |
| `document_analyze.py` | Performs perspective warping, OCR text extraction, and structural analysis. |
| `visualize_doclatyolov.py` | Visualization utility for drawing bounding boxes and verifying coordinate transforms. |
| `fasterrcnn-r50-fpn.ipynb` | Baseline experiments using Detectron2 and Faster R-CNN. |
| `fatserRcnn.ipynb` | Experimental notebook for secondary model validation. |

---

## 🧠 Models Used

* **YOLOv10 (DocLayout-YOLO):** The primary engine, optimized for document structure benchmarks.
* **Faster R-CNN (ResNet-50-FPN):** Used as a baseline for accuracy comparison in complex layouts.
* **Tesseract OCR:** Provides multilingual text extraction and orientation detection.

---

<img width="612" height="792" alt="doc_00097" src="https://github.com/user-attachments/assets/da2dc728-af78-46e2-ad0f-f5c81ab8ddd9" />

<img width="612" height="792" alt="doc_03294" src="https://github.com/user-attachments/assets/dceba677-38e3-4a89-8e4a-4ea3cb1190ec" />

## ⚙️ Installation

### 1. Environment Setup
```bash
conda create -n doclayout python=3.9
conda activate doclayout
```
### 2. Python Dependencies
```Bash

pip install torch torchvision torchaudio
pip install opencv-python pillow numpy pyyaml scikit-learn
pip install pytesseract transformers timm huggingface_hub
pip install doclayout-yolo ultralytics
```
### 3. Tesseract OCR Engine
Install Tesseract on your OS and update the path in your scripts:
Windows: Download Installer
Path Configuration: Update the following line in your .py files:
Python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

## 🗂 Dataset Preparation
The train.py script automates the conversion from COCO-style JSON to YOLO format:
Deskewing: Corrects image orientation based on corruption.severity metadata.
Mapping: Consolidates labels into target classes (text, title, images).
Splitting: Performs an 80/20 train/validation split.
Run:
```Bash
python train.py
```
## 🚀 Training
Training utilizes YOLOv10 with document-specific augmentations:
```Python
model.train(
    data="doc_dataset.yaml",
    epochs=50,
    imgsz=1024,
    batch=4,
    rotate=15.0,  # Robustness to small tilts
    shear=5.0,
    mosaic=1.0    # Handle diverse document scales
)
```
## 🔍 Usage
Single Image Inference
To process a single document with rotation correction and visualization:
```Bash
python detect_multilingual.py
```
Full Document Analysis
To extract OCR text and perform perspective-corrected structural mapping:
```Bash
python document_analyze.py
```
## 🛠 Troubleshooting
[!TIP] Issue: Bounding boxes appear shifted or misaligned.

Solution: Ensure the inverse_transform_bbox logic in visualize_doclatyolov.py matches the rotation angle used during preprocessing.

[!IMPORTANT] Issue: Tesseract fails to detect orientation.

Solution: Ensure the document has a sufficient amount of text (at least 2-3 sentences) for accurate OSD results.

## 🧪 Notebooks
fasterrcnn-r50-fpn.ipynb: Best for testing standard ResNet backbones

fatserRcnn.ipynb: Useful for calculating mAP scores against ground truth JSONs.
