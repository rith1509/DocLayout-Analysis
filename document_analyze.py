import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import json
import os
# import imutils  # <- Removed, as it was not used

from typing import List, Tuple, Dict

# ---------- Utilities ----------
def load_image(path: str):
    print(f"  [Debug] Loading image from: {path}")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Can't read image {path}")
    return img

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def order_points(pts: np.ndarray) -> np.ndarray:
    # order points (tl, tr, br, bl)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute width/height of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M, rect

def inverse_perspective_points(pts: np.ndarray, M_inv: np.ndarray) -> np.ndarray:
    # pts shape (N, 2)
    ptsf = np.array(pts, dtype="float32").reshape(-1, 1, 2)
    orig = cv2.perspectiveTransform(ptsf, M_inv)
    return orig.reshape(-1, 2)

# ---------- Preprocessing ----------
def auto_detect_document_contour(img_gray: np.ndarray) -> np.ndarray:
    print("  [Debug] Detecting document contour...")
    # edge + contour approach to find biggest quadrilateral (document)
    # Adaptive thresholding + morphology helps for low-contrast/blurry images
    blurred = cv2.GaussianBlur(img_gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("  [Debug] No contours found at all.")
        return None
        
    print(f"  [Debug] Found {len(contours)} initial contours.")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            print("  [Debug] Found 4-point polygon approximation. Using this.")
            pts = approx.reshape(4,2).astype("float32")
            return pts
            
    # fallback: bounding rect of largest contour
    print("  [Debug] No 4-point contour found. Falling back to largest contour's bounding box.")
    c = contours[0]
    x,y,w,h = cv2.boundingRect(c)
    pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype="float32")
    return pts

def unsharp_mask(image, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=0):
    print("  [Debug] Applying unsharp mask...")
    # basic sharpening to help OCR on blurred images
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

# ---------- Detection ----------
def detect_text_boxes(warped_rgb: np.ndarray) -> List[Dict]:
    print("  [Debug] Running Tesseract OCR (image_to_data)...")
    # Use pytesseract to get word/line bounding boxes
    # We'll parse pytesseract's 'tsv' output for lines and words
    data = pytesseract.image_to_data(warped_rgb, output_type=pytesseract.Output.DICT)
    n = len(data['level'])
    print(f"  [Debug] Tesseract returned {n} data items.")
    boxes = []
    for i in range(n):
        text = data['text'][i].strip()
        
        # --- START FIX ---
        # The original line caused an AttributeError because data['conf'][i] is an INT, not a string.
        # conf = float(data['conf'][i]) if data['conf'][i].isdigit() or (data['conf'][i].replace('-', '').replace('.', '').isdigit()) else -1
        
        # New, robust line:
        raw_conf = data['conf'][i]
        if isinstance(raw_conf, (int, float)):
            conf = float(raw_conf)
        elif isinstance(raw_conf, str):
            try:
                # Handle cases where it might be a string like '10.5' or '-1'
                conf = float(raw_conf)
            except ValueError:
                conf = -1.0
        else:
            conf = -1.0
        # --- END FIX ---
            
        if text == "" and conf < 0:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        boxes.append({
            "text": text,
            "conf": conf,
            "bbox": [int(x), int(y), int(w), int(h)],
            "level": data['level'][i]
        })
    print(f"  [Debug] Filtered to {len(boxes)} text boxes (words/symbols).")
    return boxes

def group_lines_to_paragraphs(text_boxes: List[Dict]) -> List[Dict]:
    print("  [Debug] Grouping text boxes into lines...")
    # group by line blocks (level 5 = word, 4 = line, 3 = paragraph depending on tesseract)
    # We'll cluster boxes by y coordinate proximity to form line-level boxes
    boxes = [b for b in text_boxes]
    if not boxes:
        print("  [Debug] No text boxes to group.")
        return []
    # convert to array with center y
    arr = []
    for b in boxes:
        x,y,w,h = b['bbox']
        arr.append((x,y,w,h,b['text'],b['conf']))
    # sort by y then x
    arr.sort(key=lambda a: (a[1], a[0])) # Fixed sorting key
    lines = []
    current = None
    for (x,y,w,h,text,conf) in arr:
        cy = y + h/2
        if current is None:
            current = {"x":x,"y":y,"w":w,"h":h,"text":text,"confs":[conf]}
        else:
            # if vertically close -> same line
            if abs((current['y']+current['h']/2) - cy) < max(10, 0.6*current['h']):
                # extend
                right = max(current['x']+current['w'], x+w)
                left = min(current['x'], x)
                top = min(current['y'], y)
                bottom = max(current['y']+current['h'], y+h)
                current['x']=left; current['y']=top; current['w']=right-left; current['h']=bottom-top
                current['text'] += " " + text
                current['confs'].append(conf)
            else:
                current['score'] = float(np.mean([c for c in current['confs'] if c>=0])) if current['confs'] else -1
                lines.append(current)
                current = {"x":x,"y":y,"w":w,"h":h,"text":text,"confs":[conf]}
    if current:
        current['score'] = float(np.mean([c for c in current['confs'] if c>=0])) if current['confs'] else -1
        lines.append(current)
    
    print(f"  [Debug] Grouped into {len(lines)} lines.")
    return lines

def detect_figures_tables(warped_gray: np.ndarray, text_masks: List[Tuple[int,int,int,int]]) -> List[Dict]:
    print("  [Debug] Detecting figures/tables by creating text mask...")
    # Create mask of text regions and remove them, then find large connected components
    mask = np.ones_like(warped_gray) * 255
    for (x,y,w,h) in text_masks:
        cv2.rectangle(mask, (x,y), (x+w, y+h), 0, -1)
    # morphological close to merge small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    mm = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # invert to get candidate regions
    cand = cv2.bitwise_not(mm)
    # threshold
    _, cand = cv2.threshold(cand, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  [Debug] Found {len(contours)} non-text contours.")
    
    h,w = warped_gray.shape[:2]
    figs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.001 * (w*h):  # skip tiny
            continue
        rx,ry,rw,rh = cv2.boundingRect(c)
        # classify table vs figure: look for many straight lines inside bbox
        roi = warped_gray[ry:ry+rh, rx:rx+rw]
        is_table = detect_table_by_line_structure(roi)
        category = "table" if is_table else "figure"
        print(f"  [Debug]   -> Found candidate: {category} (area: {area})")
        figs.append({
            "category": category,
            "bbox": [int(rx), int(ry), int(rw), int(rh)],
            "area": float(area)
        })
    # sort largest first
    figs = sorted(figs, key=lambda x: -x['area'])
    return figs

def detect_table_by_line_structure(roi_gray: np.ndarray) -> bool:
    # detect many horizontal/vertical lines using morphological operations
    h, w = roi_gray.shape[:2]
    if h < 10 or w < 10:
        return False
    # preprocess
    # Use inverted threshold: tables often have black lines on white bg
    _, thr = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # horizontal
    scale = max(1, w//40)
    horiz_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1))
    horiz = cv2.erode(thr, horiz_struct)
    horiz = cv2.dilate(horiz, horiz_struct)
    
    # vertical
    scalev = max(1, h//40)
    vert_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1,scalev))
    vert = cv2.erode(thr, vert_struct)
    vert = cv2.dilate(vert, vert_struct)
    
    # count non-zero pixels (lines)
    horiz_count = np.count_nonzero(horiz)
    vert_count = np.count_nonzero(vert)
    total_pixels = h * w
    
    # heuristics: if horizontal or vertical lines occupy > 1% of the ROI pixels
    if (horiz_count > 0.01 * total_pixels) or (vert_count > 0.01 * total_pixels):
        # print("      [Sub-Debug] Classified as TABLE")
        return True
        
    # print("      [Sub-Debug] Classified as FIGURE")
    return False

# ---------- Main pipeline ----------
def analyze_document(image_path: str) -> Dict:
    print(f"--- Starting Analysis for: {image_path} ---")
    orig = load_image(image_path)
    orig_h, orig_w = orig.shape[:2]
    print(f"Loaded image: {orig_w}x{orig_h}")
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # 1) detect document quad for perspective correction
    print("\nStep 1: Detecting Document Contour...")
    doc_pts = auto_detect_document_contour(gray)
    used_perspective = True
    if doc_pts is None:
        print("Step 1: FAILED. No contour found. Using full image as fallback.")
        # fallback: use whole image
        pts = np.array([[0,0],[orig_w-1,0],[orig_w-1,orig_h-1],[0,orig_h-1]], dtype="float32")
        doc_pts = pts
        used_perspective = False
    else:
        print("Step 1: Success. Found document contour.")

    print("\nStep 2: Applying Perspective Warp & Sharpening...")
    warped_color, M, rect = four_point_transform(orig, doc_pts)
    M_inv = np.linalg.inv(M)

    # 2) enhance / deblur (unsharp mask)
    warped_sharp = unsharp_mask(warped_color, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=5)
    warped_gray = cv2.cvtColor(warped_sharp, cv2.COLOR_BGR2GRAY)
    print("Step 2: Success. Image warped and sharpened.")

    # 3) text detection via pytesseract
    print("\nStep 3: Running OCR to Detect Text...")
    # convert to RGB for pytesseract
    warped_rgb = cv2.cvtColor(warped_sharp, cv2.COLOR_BGR2RGB)
    text_boxes = detect_text_boxes(warped_rgb)
    lines = group_lines_to_paragraphs(text_boxes)
    print(f"Step 3: Success. Found {len(lines)} text lines.")

    # Build text mask list of bounding boxes (in warped coords)
    text_masks = []
    for l in lines:
        text_masks.append((int(l['x']), int(l['y']), int(l['w']), int(l['h'])))

    # 4) detect figures/tables by removing text masks and finding large components
    print("\nStep 4: Detecting Figures & Tables...")
    figs = detect_figures_tables(warped_gray, text_masks)
    print(f"Step 4: Success. Found {len(figs)} figures/tables.")

    # 5) title detection heuristic: largest font-size line near top and centered-ish
    print("\nStep 5: Detecting Title...")
    title_candidate = None
    if lines:
        # compute line heights
        # filter top 30% of page
        page_h = warped_gray.shape[0]
        top_lines = [l for l in lines if l['y'] < 0.3 * page_h]
        if not top_lines:
            print("  [Debug] No lines in top 30%, checking first 5 lines...")
            top_lines = lines[:5]
        # choose the line with largest height (w*h), and a high avg conf
        scored = []
        for l in top_lines:
            score = l['h'] * (1 + max(0, l.get('score',0))/100.0)
            scored.append((score, l))
        
        if scored:
            scored.sort(key=lambda s:-s[0])
            best = scored[0][1]
            # check center-ish (mean x ~ center) or very large fontsize
            if best:
                center_x = best['x'] + best['w']/2
                if abs(center_x - warped_gray.shape[1]/2) < 0.4*warped_gray.shape[1] or best['h'] > 0.08*page_h:
                    title_candidate = best
                    print(f"Step 5: Success. Found title candidate: '{title_candidate['text'][:50]}...'")
                else:
                    print("  [Debug] Best line not centered or large enough to be title.")
        else:
            print("  [Debug] No lines found to score for title.")

    if not title_candidate:
        print("Step 5: No suitable title candidate found.")

    # 6) map all bboxes back to original image coordinates using inverse perspective transform
    print("\nStep 6: Mapping Coordinates back to Original Image...")
    def map_bbox_warped_to_orig(bbox):
        x,y,w,h = bbox
        corners = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype="float32")
        inv_pts = inverse_perspective_points(corners, M_inv)  # shape (4,2)
        # produce bounding rectangle (xmin,ymin,w,h) in original coordinates
        xs = inv_pts[:,0]; ys = inv_pts[:,1]
        xmin = float(np.min(xs)); ymin = float(np.min(ys))
        xmax = float(np.max(xs)); ymax = float(np.max(ys))
        return [xmin, ymin, xmax - xmin, ymax - ymin], inv_pts.tolist()

    # Compose JSON
    out = {
        "source_image": os.path.basename(image_path),
        "original_size": [int(orig_w), int(orig_h)],
        "used_perspective_transform": bool(used_perspective),
        "document_quad_in_original": rect.tolist(),  # ordered quad used for warp (tl,tr,br,bl)
        "detections": {
            "text_lines": [],
            "title": None,
            "figures_tables": []
        }
    }

    # text lines
    for l in lines:
        bbox_warp = [int(l['x']), int(l['y']), int(l['w']), int(l['h'])]
        bbox_orig, polygon = map_bbox_warped_to_orig(bbox_warp)
        out['detections']['text_lines'].append({
            "text": l['text'],
            "conf": float(l.get('score', -1)),
            "bbox_in_original": [float(round(v,2)) for v in bbox_orig],
            "bbox_polygon_in_original": [[float(round(p[0],2)), float(round(p[1],2))] for p in polygon]
        })

    if title_candidate:
        bbox_warp = [int(title_candidate['x']), int(title_candidate['y']), int(title_candidate['w']), int(title_candidate['h'])]
        bbox_orig, polygon = map_bbox_warped_to_orig(bbox_warp)
        out['detections']['title'] = {
            "text": title_candidate['text'],
            "conf": float(title_candidate.get('score', -1)),
            "bbox_in_original": [float(round(v,2)) for v in bbox_orig],
            "bbox_polygon_in_original": [[float(round(p[0],2)), float(round(p[1],2))] for p in polygon]
        }

    # figures & tables
    for f in figs:
        bbox_warp = f['bbox']
        bbox_orig, polygon = map_bbox_warped_to_orig(bbox_warp)
        out['detections']['figures_tables'].append({
            "category": f['category'],
            "bbox_in_original": [float(round(v,2)) for v in bbox_orig],
            "bbox_polygon_in_original": [[float(round(p[0],2)), float(round(p[1],2))] for p in polygon],
            "warped_bbox": f['bbox'],
            "area": f['area']
        })
    
    print("Step 6: Success. All coordinates mapped.")
    print("\n--- Analysis Complete ---")
    return out

# ---------- Visualization (NEW) ----------
def draw_annotations(image: np.ndarray, analysis_results: Dict) -> np.ndarray:
    """
    Draws the detected annotations onto the original image.
    """
    print("  [Debug] Drawing annotations on image...")
    # Define colors (B, G, R)
    COLORS = {
        "quad": (255, 0, 0),      # Blue
        "text": (0, 255, 0),      # Green
        "title": (0, 255, 255),   # Yellow
        "table": (0, 0, 255),      # Red
        "figure": (255, 0, 255)    # Magenta/Purple
    }
    
    # 1. Draw Document Quad
    if analysis_results['used_perspective_transform']:
        quad_pts = np.array(analysis_results['document_quad_in_original'], dtype=np.int32)
        cv2.polylines(image, [quad_pts.reshape(-1, 1, 2)], isClosed=True, color=COLORS['quad'], thickness=2)

    # 2. Draw Text Lines
    for line in analysis_results['detections']['text_lines']:
        poly_pts = np.array(line['bbox_polygon_in_original'], dtype=np.int32)
        cv2.polylines(image, [poly_pts.reshape(-1, 1, 2)], isClosed=True, color=COLORS['text'], thickness=1)
        
    # 3. Draw Figures & Tables
    for item in analysis_results['detections']['figures_tables']:
        poly_pts = np.array(item['bbox_polygon_in_original'], dtype=np.int32)
        category = item['category']
        color = COLORS.get(category, (255, 255, 255)) # Default to white
        
        cv2.polylines(image, [poly_pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)
        
        # Add label
        label = category.upper()
        label_pos = (int(poly_pts[0,0,0]), int(poly_pts[0,0,1]) - 10) # Top-left corner, 10px above
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    # 4. Draw Title
    title_data = analysis_results['detections']['title']
    if title_data:
        poly_pts = np.array(title_data['bbox_polygon_in_original'], dtype=np.int32)
        color = COLORS['title']
        cv2.polylines(image, [poly_pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)
        
        # Add label
        label_pos = (int(poly_pts[0,0,0]), int(poly_pts[0,0,1]) - 10) # Top-left corner, 10px above
        cv2.putText(image, "TITLE", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    print("  [Debug] Finished drawing annotations.")
    return image

# ---------- CLI ----------
if __name__ == "__main__":
    
    # --- Hard-coded path for testing ---
    # Use 'r' before the string to handle backslashes correctly
    image_path = r"D:\PS05_SHORTLIST_DATA\PS05_SHORTLIST_DATA\images\710b73ac-57b2-4a8f-9490-9d9ff40648e7.jpg"
    
    # Check if file exists before running
    if not os.path.exists(image_path):
        print(f"Error: File not found at path: {image_path}")
        print("Please check the path and try again.")
    else:
        # 1. Run Analysis
        res = analyze_document(image_path)
        
        # 2. Save JSON output
        out_path = os.path.splitext(image_path)[0] + "_doc_analysis.json"
        print(f"\nSaving JSON results to: {out_path}")
        save_json(res, out_path)
        print(f"Saved JSON -> {out_path}")
        
        # 3. Create and Save Visualization (NEW)
        print("\nGenerating visualization image...")
        vis_image = cv2.imread(image_path) # Load a fresh copy of the original image
        vis_image = draw_annotations(vis_image, res)
        
        vis_path = os.path.splitext(image_path)[0] + "_doc_visualization.jpg"
        try:
            cv2.imwrite(vis_path, vis_image)
            print(f"Saved Visualization -> {vis_path}")
        except Exception as e:
            print(f"Error saving visualization image: {e}")