import os
import json
import yaml
import time
import torch
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from doclayout_yolo import YOLOv10

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 1. CONFIGURATION ---
BASE_DIR = Path(__file__).parent
SOURCE_IMG_DIR = BASE_DIR / "train_PS05" / "train"
SOURCE_JSON_DIR = BASE_DIR / "train_PS05" / "trainJson"
OUTPUT_DATA_DIR = BASE_DIR / "document_dataset_yolo"
SUBMISSION_JSON = BASE_DIR / "submission_final.json"

# --- 2. CATEGORY MAPPING ---
CATEGORY_MAP = {1: 0, 2: 1, 5: 2}  # original → model
CLASS_NAMES = ['text', 'title', 'images']
REVERSE_MAP = {v: k for k, v in CATEGORY_MAP.items()}  # model → original

# --- 3. ROTATION HANDLING ---
def get_rotation_angle(json_path):
    """Returns clockwise rotation angle (negative) from corruption.severity."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        corruption = data.get("corruption", {})
        if corruption.get("type") == "rotate":
            severity = corruption.get("severity", 0)
            angle = -severity * 6.0  # severity 5 → -30°
            print(f"[DEBUG] {json_path.name}: rotation severity={severity}, angle={angle:.2f}°")
            return angle
    except Exception as e:
        print(f" [WARN] Failed to read corruption in {json_path.name}: {e}")
    return 0.0


def rotate_image(image, angle):
    """Rotate image by angle (positive = CCW)."""
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print(f"[DEBUG] Image rotated by {angle:.2f}° (shape={w}x{h})")
    return rotated


def inverse_transform_bbox(bbox_yolo, img_shape, angle):
    """
    Convert YOLO bbox in upright image → original rotated image.
    Returns COCO [x_min, y_min, w, h] in original space.
    """
    w, h = img_shape
    xc, yc, bw, bh = bbox_yolo

    if abs(angle) < 0.1:
        x_min = (xc - bw / 2) * w
        y_min = (yc - bh / 2) * h
        return [x_min, y_min, bw * w, bh * h]

    # Convert to corners
    x_min = (xc - bw / 2) * w
    y_min = (yc - bh / 2) * h
    x_max = (xc + bw / 2) * w
    y_max = (yc + bh / 2) * h

    corners = np.array([
        [x_min, y_min], [x_max, y_min],
        [x_max, y_max], [x_min, y_max]
    ])
    ones = np.ones((4, 1))
    corners_hom = np.hstack([corners, ones])

    # Rotate
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    corners_rot = corners_hom @ M.T

    xs, ys = corners_rot[:, 0], corners_rot[:, 1]
    x_min_r, y_min_r = xs.min(), ys.min()
    w_r, h_r = xs.max() - xs.min(), ys.max() - ys.min()

    print(f"[DEBUG] Rotating bbox by {angle:.2f}° → new [x={x_min_r:.1f}, y={y_min_r:.1f}, w={w_r:.1f}, h={h_r:.1f}]")
    return [x_min_r, y_min_r, w_r, h_r]


# --- 4. DATA PREPARATION ---
def convert_to_yolo(json_path, img_width, img_height):
    """Convert COCO annotations → YOLO format (in original image space)."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        yolo_labels = []
        for ann in data['annotations']:
            cat_id = ann['category_id']
            if cat_id in CATEGORY_MAP:
                x_min, y_min, w, h = ann['bbox']
                xc = (x_min + w / 2) / img_width
                yc = (y_min + h / 2) / img_height
                wn = w / img_width
                hn = h / img_height
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= wn <= 1 and 0 <= hn <= 1):
                    print(f"[WARN] Invalid normalized box in {json_path.name}: {xc:.3f},{yc:.3f},{wn:.3f},{hn:.3f}")
                    continue
                if w <= 0 or h <= 0:
                    print(f"[WARN] Zero-size bbox skipped in {json_path.name}")
                    continue
                yolo_labels.append(f"{CATEGORY_MAP[cat_id]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        print(f"[DEBUG] {json_path.name}: {len(yolo_labels)} valid boxes converted.")
        return yolo_labels
    except Exception as e:
        print(f" [ERROR] Failed to convert {json_path.name}: {e}")
        return []


def process_split(json_file_list, source_img_dir, split_name):
    print(f"\n--- Processing {split_name} split ({len(json_file_list)} files) ---")
    start_time = time.time()
    img_dir = OUTPUT_DATA_DIR / 'images' / split_name
    lbl_dir = OUTPUT_DATA_DIR / 'labels' / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    processed, total_boxes = 0, 0
    total_files = len(json_file_list)

    for i, json_path in enumerate(json_file_list, 1):
        base_name = json_path.stem
        img_candidates = [source_img_dir / f"{base_name}{ext}" for ext in ['.png', '.jpg', '.jpeg']]
        img_path = next((p for p in img_candidates if p.exists()), None)
        if not img_path:
            print(f"[WARN] Missing image for {base_name}")
            continue

        try:
            # Get rotation angle
            angle = get_rotation_angle(json_path)
            deskew_angle = -angle

            # Load + rotate image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[ERROR] Cannot read {img_path.name}")
                continue
            upright_img = rotate_image(img, deskew_angle)
            upright_path = img_dir / img_path.name
            cv2.imwrite(str(upright_path), upright_img)

            # Convert annotations
            orig_h, orig_w = img.shape[:2]
            yolo_labels = convert_to_yolo(json_path, orig_w, orig_h)
            if not yolo_labels:
                continue

            label_lines = []
            for line in yolo_labels:
                parts = list(map(float, line.split()))
                cls_id = int(parts[0])
                bbox_yolo = parts[1:]
                bbox_coco = inverse_transform_bbox(bbox_yolo, (orig_w, orig_h), deskew_angle)
                x_min, y_min, w, h = bbox_coco

                up_h, up_w = upright_img.shape[:2]
                xc_new = (x_min + w / 2) / up_w
                yc_new = (y_min + h / 2) / up_h
                wn_new = w / up_w
                hn_new = h / up_h

                if 0 <= xc_new <= 1 and 0 <= yc_new <= 1 and wn_new > 0 and hn_new > 0:
                    label_lines.append(f"{cls_id} {xc_new:.6f} {yc_new:.6f} {wn_new:.6f} {hn_new:.6f}")
                else:
                    print(f"[DEBUG] Discarded out-of-bound bbox after rotation in {json_path.name}")

            if label_lines:
                (lbl_dir / f"{base_name}.txt").write_text("\n".join(label_lines))
                total_boxes += len(label_lines)
                processed += 1

        except Exception as e:
            print(f" [ERROR] Failed to process {img_path.name}: {e}")

        if i % 100 == 0 or i == total_files:
            percent = (i / total_files) * 100
            print(f"   Progress: {i}/{total_files} ({percent:.1f}%)")

    elapsed = time.time() - start_time
    print(f"Finished {split_name}: {processed} images, {total_boxes} boxes in {elapsed:.1f}s")
    return total_boxes, json_file_list


def prepare_dataset():
    print("\n=== Preparing Dataset ===")
    start_time = time.time()

    json_files = list(SOURCE_JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSONs in {SOURCE_JSON_DIR}")
        return False, None, None

    print(f"[INFO] Found {len(json_files)} total JSON files.")
    train_files, val_files = train_test_split(json_files, test_size=0.1, random_state=42)
    print(f"[INFO] Train: {len(train_files)}, Val: {len(val_files)}")

    process_split(train_files, SOURCE_IMG_DIR, 'train')
    _, val_json_list = process_split(val_files, SOURCE_IMG_DIR, 'val')

    elapsed = time.time() - start_time
    print(f"[INFO] Dataset preparation completed in {elapsed:.1f}s")
    return True, val_files, val_json_list


def create_yaml_file():
    print("\n=== Creating Dataset YAML ===")
    yaml_path = OUTPUT_DATA_DIR / 'doc_dataset.yaml'
    content = {
        'path': str(OUTPUT_DATA_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(content, f, default_flow_style=False)
    print(f"[INFO] YAML created at {yaml_path}")
    return yaml_path


def run_training(yaml_path):
    print("\n=== Starting Training ===")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Training device: {device}")
    # 2. In your code, replace the from_pretrained line with:
    model = YOLOv10("yolov10n.pt")  # Direct load
    model.to(device)

    start_time = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=1024,
        batch=4,
        project=str(BASE_DIR / 'runs'),
        name='rotated_finetune',
        rotate=15.0,
        shear=5.0,
        mosaic=1.0,
        seed=42
    )
    elapsed = time.time() - start_time
    best = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"[INFO] Training complete in {elapsed/60:.2f} min → Best model: {best}")
    return best


def generate_submission(model_path, val_json_files):
    print("\n=== Generating Submission ===")
    model = YOLOv10(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"[INFO] Inference using device: {device}")

    val_img_dir = OUTPUT_DATA_DIR / 'images' / 'val'
    submission = []
    total_files = len(val_json_files)
    start_time = time.time()

    for i, json_path in enumerate(val_json_files, 1):
        base_name = json_path.stem
        upright_img_path = next(val_img_dir.glob(f"{base_name}.*"), None)
        orig_img_path = next(SOURCE_IMG_DIR.glob(f"{base_name}.*"), None)
        if not upright_img_path or not orig_img_path:
            print(f"[WARN] Missing upright or original image for {base_name}")
            continue

        with Image.open(orig_img_path) as img:
            orig_w, orig_h = img.size
        angle = get_rotation_angle(json_path)

        results = model(str(upright_img_path), imgsz=1024, conf=0.25)[0]

        bboxes = []
        for box in results.boxes:
            cls_id_model = int(box.cls.item())
            orig_cls_id = REVERSE_MAP.get(cls_id_model, cls_id_model)
            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]

            up_h, up_w = results.orig_shape
            xc = ((x1 + x2) / 2) / up_w
            yc = ((y1 + y2) / 2) / up_h
            bw = (x2 - x1) / up_w
            bh = (y2 - y1) / up_h

            bbox_orig = inverse_transform_bbox([xc, yc, bw, bh], (up_w, up_h), angle)

            bboxes.append({
                "category_id": orig_cls_id,
                "bbox": [round(x, 1) for x in bbox_orig],
                "score": round(conf, 3)
            })

        submission.append({
            "image_id": orig_img_path.name,
            "width": orig_w,
            "height": orig_h,
            "bboxes": bboxes
        })

        if i % 50 == 0 or i == total_files:
            percent = (i / total_files) * 100
            print(f"   Inference progress: {i}/{total_files} ({percent:.1f}%)")

    with open(SUBMISSION_JSON, 'w') as f:
        json.dump(submission, f, indent=2)
    elapsed = time.time() - start_time
    print(f"[INFO] Submission JSON saved at {SUBMISSION_JSON} ({elapsed:.1f}s)")
    return SUBMISSION_JSON


# --- MAIN ---
def main():
    print("==========================================")
    print(" DocLayout-YOLO + Rotation Handling (DEBUG MODE) ")
    print("==========================================")

    total_start = time.time()

    success, val_files, val_json_list = prepare_dataset()
    if not success:
        print("[FATAL] Dataset preparation failed.")
        return

    yaml_path = create_yaml_file()
    best_model = run_training(yaml_path)
    generate_submission(best_model, val_json_list)

    total_elapsed = time.time() - total_start
    print(f"\n[ALL DONE] Total pipeline finished in {total_elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    main()
