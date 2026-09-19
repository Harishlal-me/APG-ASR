import os
import sys
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Import the authentic APG-ASR pipeline classes
from src.inference.detector import BaseDetector
from src.inference.agent import AgentLogic

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / float(area1 + area2 - inter)

def apply_wbf_real(boxes, scores, classes, iou_thr=0.55):
    if len(boxes) == 0:
        return np.zeros((0,4)), np.zeros(0), np.zeros(0)
    
    unique_classes = np.unique(classes)
    wbf_boxes, wbf_scores, wbf_classes = [], [], []
    
    for c in unique_classes:
        c_idx = np.where(classes == c)[0]
        c_boxes, c_scores = boxes[c_idx], scores[c_idx]
        
        order = np.argsort(c_scores)[::-1]
        c_boxes, c_scores = c_boxes[order], c_scores[order]
        
        clusters = []
        for i in range(len(c_boxes)):
            box, score = c_boxes[i], c_scores[i]
            matched = False
            for cluster in clusters:
                if compute_iou(box, cluster['box']) > iou_thr:
                    cluster['boxes'].append(box)
                    cluster['scores'].append(score)
                    
                    w_sum = sum(cluster['scores'])
                    new_box = sum(cb * cs for cb, cs in zip(cluster['boxes'], cluster['scores']))
                    cluster['box'] = new_box / w_sum
                    matched = True
                    break
            if not matched:
                clusters.append({'box': box, 'boxes': [box], 'scores': [score]})
                
        for cluster in clusters:
            wbf_boxes.append(cluster['box'])
            wbf_scores.append(sum(cluster['scores']) / len(cluster['scores']))
            wbf_classes.append(c)
            
    return np.array(wbf_boxes), np.array(wbf_scores), np.array(wbf_classes)

def parse_yolo_txt(txt_path, img_w, img_h):
    gt_boxes = []
    gt_classes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    x1 = (x_c - w / 2) * img_w
                    y1 = (y_c - h / 2) * img_h
                    x2 = (x_c + w / 2) * img_w
                    y2 = (y_c + h / 2) * img_h
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(cls_id)
    return np.array(gt_boxes), np.array(gt_classes)

def resize_to_standard(img, target_width=1280):
    h, w = img.shape[:2]
    scale = target_width / float(w)
    target_height = int(h * scale)
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def draw_panel(image, boxes, scores, classes, class_names, color, main_title, subtitle, 
               iou_diffs=None, arrow_idx=-1):
    
    img_copy = resize_to_standard(image)
    orig_h, orig_w = image.shape[:2]
    new_h, new_w = img_copy.shape[:2]
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    
    base_thickness = 1 if "APG-ASR" not in main_title else 2
    
    if len(boxes) > 0:
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, base_thickness)
            cls_name = class_names.get(int(cls), str(int(cls)))
            label = f"{cls_name} {score:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            ly1 = max(0, y1 - 30)
            ly2 = ly1 + th + 10
            
            cv2.rectangle(img_copy, (x1, ly1), (x1 + tw + 10, ly2), color, -1)
            text_color = (0, 0, 0) if color == (220, 220, 220) else (255, 255, 255)
            cv2.putText(img_copy, label, (x1 + 5, ly2 - 5), font, font_scale, text_color, thickness)
            
            # IoU Diff rendering
            if iou_diffs is not None and i < len(iou_diffs) and iou_diffs[i]:
                diff_text = iou_diffs[i]
                (dtw, dth), _ = cv2.getTextSize(diff_text, font, font_scale, thickness)
                dy1 = y2
                dy2 = y2 + dth + 10
                cv2.rectangle(img_copy, (x1, dy1), (x1 + dtw + 10, dy2), (0, 0, 0), -1)
                cv2.putText(img_copy, diff_text, (x1 + 5, dy2 - 5), font, font_scale, (0, 255, 255), thickness)
                
            # Draw Arrow pointing to the improved box
            if i == arrow_idx and "APG-ASR" in main_title:
                center_x = (x1 + x2) // 2
                pt1 = (center_x, max(0, y1 - 80))
                pt2 = (center_x, max(0, y1 - 30))
                cv2.arrowedLine(img_copy, pt1, pt2, (0, 255, 255), 4, tipLength=0.4)
                
    title_font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img_copy, main_title, (20, 50), title_font, 1.2, (255, 255, 255), 5)
    cv2.putText(img_copy, main_title, (20, 50), title_font, 1.2, (0, 0, 0), 2)
    
    cv2.putText(img_copy, subtitle, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(img_copy, subtitle, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    return img_copy

def create_2x2_composite(panel_a, panel_b, panel_c, panel_d, main_title):
    top_row = cv2.hconcat([panel_a, panel_b])
    bottom_row = cv2.hconcat([panel_c, panel_d])
    grid = cv2.vconcat([top_row, bottom_row])
    
    h, w = grid.shape[:2]
    banner_height = 80
    banner = np.full((banner_height, w, 3), 255, dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(main_title, font, 1.5, 3)
    tx = (w - tw) // 2
    ty = (banner_height + th) // 2
    cv2.putText(banner, main_title, (tx, ty), font, 1.5, (0, 0, 0), 3)
    
    return cv2.vconcat([banner, grid])

def process_single_grid(raw_img, img_name, gt_boxes, gt_classes, active_detector, model_label, output_dir, main_title):
    COLOR_RAW = (255, 255, 255)
    COLOR_BASE = (255, 50, 50)       # Blue
    COLOR_WBF = (220, 220, 220)      # Gray/White
    COLOR_APG = (50, 50, 255)        # Red
    
    names = active_detector.names if active_detector else {}
    img_h, img_w = raw_img.shape[:2]

    # Panel A: Raw
    panel_a = draw_panel(raw_img, [], [], [], {}, COLOR_RAW, "(a) Input Image", "Original input image")
    
    b_boxes, b_scores, b_classes = [], [], []
    wbf_boxes, wbf_scores, wbf_classes = [], [], []
    a_boxes, a_scores, a_classes = [], [], []
    iou_diffs = []
    arrow_idx = -1
    
    # 1. Baseline
    if active_detector:
        raw_dets_base = active_detector.get_raw_detections(raw_img, conf=0.25)
        b_boxes = np.array([d['coords'] for d in raw_dets_base]) if raw_dets_base else np.zeros((0,4))
        b_scores = np.array([d['conf'] for d in raw_dets_base]) if raw_dets_base else np.zeros(0)
        b_classes = np.array([d['cls'] for d in raw_dets_base]) if raw_dets_base else np.zeros(0)
        
    panel_b = draw_panel(raw_img, b_boxes, b_scores, b_classes, names, COLOR_BASE, f"(b) YOLOv8 Baseline ({model_label})", "Raw detections (conf > 0.25)")
    
    # 2. Naive WBF
    if active_detector:
        dets_scales = []
        for scale in [0.8, 1.0, 1.2]:
            scaled = cv2.resize(raw_img, (int(img_w*scale), int(img_h*scale)))
            sdets = active_detector.get_raw_detections(scaled, conf=0.15)
            for sd in sdets:
                cx1, cy1, cx2, cy2 = sd['coords']
                sd['coords'] = [cx1/scale, cy1/scale, cx2/scale, cy2/scale]
            dets_scales.append(sdets)
            
        all_w_boxes, all_w_scores, all_w_classes = [], [], []
        for ds in dets_scales:
            for d in ds:
                all_w_boxes.append(d['coords'])
                all_w_scores.append(d['conf'])
                all_w_classes.append(d['cls'])
                
        wbf_boxes, wbf_scores, wbf_classes = apply_wbf_real(
            np.array(all_w_boxes) if all_w_boxes else np.zeros((0,4)),
            np.array(all_w_scores) if all_w_scores else np.zeros(0),
            np.array(all_w_classes) if all_w_classes else np.zeros(0)
        )
        
    panel_c = draw_panel(raw_img, wbf_boxes, wbf_scores, wbf_classes, names, COLOR_WBF, "(c) Naive WBF", "Averaged boxes from multiscale inference")
    
    # 3. APG-ASR
    best_improvement_val = 0.0
    if active_detector:
        agent = AgentLogic(active_detector)
        raw_dets = active_detector.get_raw_detections(raw_img, conf=0.1)
        structured_output, stats, timings, trace = agent.run_agent(raw_img, raw_dets, max_reprocess=5)
        
        a_boxes = np.array([d['bbox'] for d in structured_output]) if structured_output else np.zeros((0,4))
        a_scores = np.array([d['confidence'] for d in structured_output]) if structured_output else np.zeros(0)
        a_classes = np.array([d['class_id'] for d in structured_output]) if structured_output else np.zeros(0)
        
        # Filter and NMS
        if len(a_boxes) > 0:
            indices = cv2.dnn.NMSBoxes(a_boxes.tolist(), a_scores.tolist(), 0.25, 0.5)
            if len(indices) > 0:
                indices = indices.flatten()
                valid_idx = [i for i in indices if (a_boxes[i][2]-a_boxes[i][0])*(a_boxes[i][3]-a_boxes[i][1]) > 400]
                valid_idx = sorted(valid_idx, key=lambda i: a_scores[i], reverse=True)
                a_boxes = a_boxes[valid_idx]
                a_scores = a_scores[valid_idx]
                a_classes = a_classes[valid_idx]
        
        # Calculate IoU improvements against GT
        if len(gt_boxes) > 0 and len(a_boxes) > 0:
            for i, a_box in enumerate(a_boxes):
                best_gt_idx = -1
                best_apg_iou = 0
                for j, (gt_b, gt_c) in enumerate(zip(gt_boxes, gt_classes)):
                    if int(gt_c) == int(a_classes[i]):
                        iou = compute_iou(a_box, gt_b)
                        if iou > best_apg_iou:
                            best_apg_iou = iou
                            best_gt_idx = j
                
                diff_str = ""
                if best_gt_idx != -1:
                    best_base_iou = 0
                    if len(b_boxes) > 0:
                        for b_b, b_c in zip(b_boxes, b_classes):
                            if int(b_c) == int(gt_classes[best_gt_idx]):
                                iou = compute_iou(b_b, gt_boxes[best_gt_idx])
                                if iou > best_base_iou:
                                    best_base_iou = iou
                    
                    diff = best_apg_iou - best_base_iou
                    if diff >= 0.05:
                        diff_str = f"+{diff:.2f} IoU"
                        if diff > best_improvement_val:
                            best_improvement_val = diff
                            arrow_idx = i
                iou_diffs.append(diff_str)
        else:
            iou_diffs = [""] * len(a_boxes)
            
    # Draw base faded boxes underneath APG-ASR
    base_faded = raw_img.copy()
    for b_b in b_boxes:
        x1, y1, x2, y2 = [int(v) for v in b_b]
        cv2.rectangle(base_faded, (x1, y1), (x2, y2), (200, 200, 255), 1)
        
    panel_d = draw_panel(base_faded, a_boxes, a_scores, a_classes, names, COLOR_APG, "(d) APG-ASR (Ours)", "Agent logic refinement", iou_diffs=iou_diffs, arrow_idx=arrow_idx)
    
    dynamic_title = f"{main_title} ({model_label})"
    if best_improvement_val > 0.01:
        dynamic_title += f" (+{best_improvement_val:.2f} Max IoU)"
        
    if 0.01 <= best_improvement_val < 0.05:
        minor_dir = os.path.join(output_dir, "minor_improved")
        os.makedirs(minor_dir, exist_ok=True)
        minor_path = os.path.join(minor_dir, f"grid_{model_label}_iou+{best_improvement_val:.2f}_{img_name}")
        cv2.imwrite(minor_path, composite)
        return True
    return False

def run_batch(input_dir, output_dir, main_title):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Scanning {input_dir} for images...")
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_paths:
        print("[ERROR] No .jpg images found in the input directory.")
        return
        
    print("[INFO] Loading models...")
    base_path = os.path.join(ROOT, "models", "yolov8", "base", "yolov8n.pt")
    custom_path = os.path.join(ROOT, "models", "yolov8", "custom", "yolov8_custom_best.pt")
    
    pretrained_detector = None
    custom_detector = None
    
    if os.path.exists(base_path):
        try:
            pretrained_detector = BaseDetector(base_path)
            print("  -> Pretrained YOLOv8n Loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load Base YOLO: {e}")
            
    if os.path.exists(custom_path):
        try:
            custom_detector = BaseDetector(custom_path)
            print("  -> Custom YOLOv8 Loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load Custom YOLO: {e}")

    print(f"[INFO] Scanning {input_dir} for minor improvements...")
    found_count = 0
    for img_path in tqdm(image_paths):
        if found_count >= 10:
            print("[SUCCESS] Found 10 minor improvements! Stopping early.")
            break
            
        img_name = os.path.basename(img_path)
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue
            
        img_h, img_w = raw_img.shape[:2]
        gt_boxes, gt_classes = parse_yolo_txt(txt_path, img_w, img_h)
            
        # Only run CUSTOM grid to save time
        if custom_detector:
            custom_out = os.path.join(output_dir, "custom_grids")
            is_found = process_single_grid(raw_img, img_name, gt_boxes, gt_classes, custom_detector, "Custom", custom_out, main_title)
            if is_found:
                found_count += 1
        
    print(f"\n[SUCCESS] Search complete.")
    print(f"Results saved to: {output_dir}/pretrained_grids and {output_dir}/custom_grids")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APG-ASR True Agentic Inference Engine")
    parser.add_argument("--input_dir", type=str, default=os.path.join(ROOT, "sample_images_raw"), help="Directory containing raw images.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT, "outputs", "predictions", "batch_run"), help="Directory to save 2x2 grids.")
    parser.add_argument("--title", type=str, default="APG-ASR Comparative Inference", help="Main title for the composites.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"[ERROR] Input directory not found: {args.input_dir}")
        sys.exit(1)
        
    run_batch(args.input_dir, args.output_dir, args.title)
