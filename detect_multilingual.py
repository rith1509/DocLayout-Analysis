import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from doclayout_yolo import YOLOv10
import logging
import re

# === 1. LIBRARIES ===
# Make sure you have run 'pip install transformers timm pytesseract'
from huggingface_hub import hf_hub_download
import pytesseract

#
# === 2. TESSERACT CONFIG ===
# This is the default path on Windows.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
#
# This is the crucial part for multi-language orientation detection
#
LANGUAGE_STRING = "eng+spa+fra+ara+hin+urd+sin"
#

# === CONFIG ===
BASE_DIR = Path(__file__).parent

# The original model (Our One Expert)
YOLO_MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
YOLO_MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"


# === 3. EDIT THIS PATH ===
# Point this to *ANY* single image file, in *ANY* rotation.
# Using the Arabic-script image from your D: drive.
IMAGE_TO_TEST_STRING = r"D:\PS05_SHORTLIST_DATA\PS05_SHORTLIST_DATA\images\0a1d6e43-811c-4a2e-83f1-33ad183b4d84.jpg" 
#
# === 4. (OPTIONAL) EDIT THIS PATH ===
#
OUTPUT_IMAGE_PATH_STRING = "multilingual_detection_result.png"

# --- (Internal paths, do not edit) ---
IMAGE_TO_TEST = Path(IMAGE_TO_TEST_STRING)
OUTPUT_IMAGE_PATH = BASE_DIR / OUTPUT_IMAGE_PATH_STRING

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# === MAPPING & COLORS ===
# These are the *actual* classes from the original model
# We will detect: text, title, table, figure
CATEGORY_NAMES = {
    0: 'text',
    1: 'title',
    3: 'table',
    4: 'figure'
}
COLOR_MAP = {
    0: (255, 0, 0),    # Blue for 'text'
    1: (0, 255, 0),    # Green for 'title'
    3: (255, 0, 255),  # Magenta for 'table'
    4: (0, 0, 255)     # Red for 'figure'
}

# === ROTATION & FUSION HELPERS ===

def get_orientation(cv2_image) -> float:
    """Finds the 0, 90, 180, or 270 degree orientation."""
    log.info("Step 2: Detecting page orientation (multilingual)...")
    try:
        # This is the new, crucial line
        osd = pytesseract.image_to_osd(cv2_image, lang=LANGUAGE_STRING)
        
        angle = int(re.search(r'(?<=Rotate: )\d+', osd).group(0))
        log.info(f"Tesseract OSD detected orientation: {angle} degrees")
        return angle
    except Exception as e:
        log.warning(f"Tesseract OSD failed: {e}. Assuming 0 angle.")
        return 0.0

def get_skew_angle(cv2_image) -> float:
    """Finds the minor skew angle of a document image."""
    log.info("Step 3: Detecting fine-tune skew angle...")
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log.warning("No contours found for skew. Assuming 0 angle.")
        return 0.0
    all_points = np.concatenate(contours)
    rect = cv2.minAreaRect(all_points)
    angle = rect[2]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    log.info(f"Skew angle detected: {angle:.2f} degrees")
    return angle

def rotate_image(image, angle):
    """Rotates an image (OpenCV) by a given angle."""
    if abs(angle) < 0.1:
        return image
    log.info(f"Applying rotation of {angle:.2f} degrees...")
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# === MONKEY-PATCH FOR YOLO ===
def apply_yolo_patch():
    log.info("Applying comprehensive monkey-patch for YOLO loading...")
    try:
        import sys
        import ultralytics.nn.tasks
        from doclayout_yolo.models.yolov10.model import YOLOv10DetectionModel
        sys.modules['ultralytics.nn.tasks'].YOLOv10DetectionModel = YOLOv10DetectionModel
        import ultralytics.nn.modules.block, doclayout_yolo.nn.modules.block as d_block
        import ultralytics.nn.modules.conv, doclayout_yolo.nn.modules.conv as d_conv
        import ultralytics.nn.modules.head, doclayout_yolo.nn.modules.head as d_head
        def patch_module(u_mod, d_mod):
            for attr in dir(d_mod):
                if not attr.startswith('__') and isinstance(getattr(d_mod, attr), type):
                    setattr(u_mod, attr, getattr(d_mod, attr))
        patch_module(ultralytics.nn.modules.block, d_block)
        patch_module(ultralytics.nn.modules.conv, d_conv)
        patch_module(ultralytics.nn.modules.head, d_head)
        log.info("YOLO monkey-patch applied successfully.")
    except ImportError as e: log.error(f"Failed to apply YOLO monkey-patch: {e}"); raise

# === MAIN VISUALIZATION FUNCTION ===
def run_multilingual_detection():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")

    # --- 1. Load the Single Expert Model (with patch) ---
    apply_yolo_patch()
    try:
        log.info(f"Loading original model from {YOLO_MODEL_REPO}...")
        local_original_path = hf_hub_download(
            repo_id=YOLO_MODEL_REPO,
            filename=YOLO_MODEL_FILENAME
        )
        model = YOLOv10(local_original_path)
        model.to(device)
        log.info("Language-Agnostic YOLO expert loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load original model: {e}")
        return

    # --- 2. Load & FULLY Correct Image (Orientation + Skew) ---
    log.info(f"Step 1: Loading image: {IMAGE_TO_TEST}")
    if not IMAGE_TO_TEST.exists():
        log.error(f"Image file not found! Check path: {IMAGE_TO_TEST}"); return
    img_cv = cv2.imread(str(IMAGE_TO_TEST))
    if img_cv is None:
        log.error(f"Failed to read image file: {IMAGE_TO_TEST}"); return
    
    # --- THIS IS THE NEW PIPELINE ---
    # 1. Fix 90/180/270 degree rotation
    orientation_angle = get_orientation(img_cv)
    img_oriented = rotate_image(img_cv, -orientation_angle) # Tesseract's angle is CW, rotate CCW
    
    # 2. Fix minor 1-5 degree skew
    skew_angle = get_skew_angle(img_oriented)
    img_upright_cv = rotate_image(img_oriented, skew_angle)
    # --- END NEW PIPELINE ---
    
    log.info("Step 4: Image loaded and fully corrected.")

    # --- 3. Run Inference ---
    log.info("Step 5: Running Language-Agnostic model...")
    results = model(img_upright_cv, imgsz=1024, conf=0.25)[0]
    log.info(f"Model found {len(results.boxes)} objects.")

    # --- 4. Draw Final Visualizations ---
    output_img = img_upright_cv.copy()
    for box in results.boxes:
        cls_id = int(box.cls.item())
        
        # We only draw the classes we care about
        if cls_id in CATEGORY_NAMES:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
            conf = float(box.conf.item())
            
            color = COLOR_MAP.get(cls_id)
            label = CATEGORY_NAMES.get(cls_id)
            label_str = f"{label}: {conf:.2f}"
            
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness=2)
            (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output_img, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1 - 5), color, -1)
            cv2.putText(output_img, label_str, (x1 + 2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=2)

    cv2.imwrite(str(OUTPUT_IMAGE_PATH), output_img)
    log.info(f"Ensemble visualization saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    # Check if Tesseract is installed before running
    if not Path(TESSERACT_CMD).exists():
        log.error("="*50)
        log.error("TESSERACT NOT FOUND!")
        log.error(f"The script expected Tesseract at: {TESSERACT_CMD}")
        log.error("Please install Tesseract-OCR from 'https://github.com/UB-Mannheim/tesseract/wiki'")
        log.error("Or update the TESSERACT_CMD path in this script.")
        log.error("="*50)
    else:
        try:
            run_multilingual_detection()
        except Exception as e:
            log.error(f"An unhandled error occurred: {e}")
            import traceback
            traceback.print_exc()