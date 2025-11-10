# infer_single.py
import os
import cv2
import numpy as np
from pathlib import Path
from doclayout_yolo import YOLOv10
import torch

# ============================= CONFIG =============================
MODEL_PATH = r"C:\Users\Dell\Desktop\dpiit_ps5_yolov10\runs\finetune\weights\best.pt"
IMAGE_PATH = r"D:\PS05_SHORTLIST_DATA\PS05_SHORTLIST_DATA\images\3fe6ed8e-63c6-462b-9ea7-2198241af759.jpg"

# Optional: path to JSON (same name, .json) for rotation info
JSON_PATH = Path(IMAGE_PATH).with_suffix('.json')
JSON_PATH = JSON_PATH if JSON_PATH.exists() else None

OUTPUT_DIR = Path(IMAGE_PATH).parent
RESULT_PATH = OUTPUT_DIR / f"result_{Path(IMAGE_PATH).name}"

CLASS_NAMES = ['text', 'title', 'images']
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Red, Blue
# =================================================================


def get_rotation_angle(json_path):
    if not json_path:
        return 0.0
    try:
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        corruption = data.get("corruption", {})
        if corruption.get("type") == "rotate":
            severity = corruption.get("severity", 0)
            angle = -severity * 6.0
            print(f"[INFO] Rotation detected: {angle:.1f}° (severity {severity})")
            return angle
    except Exception as e:
        print(f"[WARN] Could not read rotation from JSON: {e}")
    return 0.0


def rotate_image(image, angle):
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def inverse_transform_bbox(bbox_yolo, orig_size, angle):
    """Transform bbox from upright (deskewed) image back to original rotated image."""
    orig_w, orig_h = orig_size
    xc, yc, bw, bh = bbox_yolo

    if abs(angle) < 0.1:
        x_min = (xc - bw / 2) * orig_w
        y_min = (yc - bh / 2) * orig_h
        return [x_min, y_min, bw * orig_w, bh * orig_h]

    # Convert to corner points
    x_min = (xc - bw / 2) * orig_w
    y_min = (yc - bh / 2) * orig_h
    x_max = (xc + bw / 2) * orig_w
    y_max = (yc + bh / 2) * orig_h

    corners = np.array([
        [x_min, y_min], [x_max, y_min],
        [x_max, y_max], [x_min, y_max]
    ])
    ones = np.ones((4, 1))
    corners_hom = np.hstack([corners, ones])

    center = (orig_w / 2, orig_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    corners_rot = corners_hom @ M.T

    xs, ys = corners_rot[:, 0], corners_rot[:, 1]
    x_min_r, y_min_r = xs.min(), ys.min()
    w_r = xs.max() - x_min_r
    h_r = ys.max() - y_min_r
    return [x_min_r, y_min_r, w_r, h_r]


def main():
    print("Loading model...")
    model = YOLOv10(MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on {device}")

    print(f"Loading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    orig_h, orig_w = img.shape[:2]
    print(f"Original size: {orig_w}×{orig_h}")

    # === Handle rotation ===
    angle = get_rotation_angle(JSON_PATH)        # clockwise (negative)
    deskew_angle = -angle                         # to upright (CCW)

    upright_img = rotate_image(img, deskew_angle)
    up_h, up_w = upright_img.shape[:2]

    # === Inference on upright image ===
    temp_path = OUTPUT_DIR / "temp_upright.jpg"
    cv2.imwrite(str(temp_path), upright_img)

    results = model(str(temp_path), imgsz=1024, conf=0.25)[0]
    os.remove(temp_path)

    # === Visualization on ORIGINAL image ===
    vis_img = img.copy()

    print(f"Found {len(results.boxes)} detections.")
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        if cls_id >= len(CLASS_NAMES):
            continue

        # YOLO format in upright image
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        xc = ((x1 + x2) / 2) / up_w
        yc = ((y1 + y2) / 2) / up_h
        bw = (x2 - x1) / up_w
        bh = (y2 - y1) / up_h

        # Transform back to original rotated image
        bbox_orig = inverse_transform_bbox([xc, yc, bw, bh], (orig_w, orig_h), angle)
        x_min, y_min, w, h = [int(round(v)) for v in bbox_orig]

        # Clip to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(orig_w, x_min + w)
        y_max = min(orig_h, y_min + h)

        # Draw
        color = COLORS[cls_id]
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), color, 2)

        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (x_min, y_min - th - baseline), (x_min + tw, y_min), color, -1)
        cv2.putText(vis_img, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # === Save result ===
    cv2.imwrite(str(RESULT_PATH), vis_img)
    print(f"Result saved: {RESULT_PATH}")

    # Optional: show image
    # cv2.imshow("Result", vis_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()