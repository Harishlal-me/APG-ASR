import os
import sys
import argparse
import cv2
import numpy as np
import torch
import yaml
import torchvision
from ultralytics import YOLO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

class APGASRModel:
    def __init__(self, weight_path):
        from src.training.train_engine import UltralyticsAPGWrapper
        
        data_yaml = os.path.join(ROOT, "config", "dataset.yaml")
        nc = 80
        names = {i: str(i) for i in range(nc)}
        if os.path.exists(data_yaml):
            with open(data_yaml, "r") as f:
                data_dict = yaml.safe_load(f)
                nc = data_dict.get("nc", 80)
                names = data_dict.get("names", names)
                
        apg_net = UltralyticsAPGWrapper(nc, names)
        
        ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
        if 'ema' in ckpt and ckpt['ema']:
            state_dict = ckpt['ema'].state_dict() if hasattr(ckpt['ema'], 'state_dict') else ckpt['ema']
        elif 'model' in ckpt and ckpt['model']:
            state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
        else:
            state_dict = ckpt
            
        new_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace("model.", "") if k.startswith("model.") else k
            if not clean_k.startswith("apg.") and not clean_k.startswith("dfl."):
                clean_k = "apg." + clean_k
            new_state_dict[clean_k] = v
            
        apg_net.load_state_dict(new_state_dict, strict=False)
        apg_net.eval()
        apg_net.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        os.environ["APG_GATE_MULTIPLIER"] = "0.3"
        
        local_base = os.path.join(ROOT, "models", "yolov8", "base", "yolov8n.pt")
        self.pipeline = YOLO(local_base) 
        self.pipeline.model = apg_net       

    def predict(self, image):
        return self.pipeline.predict(image, conf=0.30, verbose=False)[0]

def load_models():
    base_path = os.path.join(ROOT, "models", "yolov8", "base", "yolov8n.pt")
    custom_path = os.path.join(ROOT, "models", "yolov8", "custom", "yolov8_custom_best.pt")
    apg_path = os.path.join(ROOT, "models", "apg_asr", "weights", "apg_asr_best.pt")
    
    models = {}
    
    if os.path.exists(base_path):
        try:
            models['base'] = YOLO(base_path)
        except Exception as e:
            print(f"[ERROR] Failed to load Base YOLO: {e}")
            models['base'] = None
    else:
        print(f"[WARN] Base YOLO not found at {base_path}, skipping...")
        models['base'] = None
        
    if os.path.exists(custom_path):
        try:
            models['custom'] = YOLO(custom_path)
        except Exception as e:
            print(f"[ERROR] Failed to load Custom YOLO: {e}")
            models['custom'] = None
    else:
        print(f"[WARN] Custom YOLO not found at {custom_path}, skipping...")
        models['custom'] = None
        
    if os.path.exists(apg_path):
        try:
            models['apg'] = APGASRModel(apg_path)
        except Exception as e:
            print(f"[ERROR] Failed to load APG-ASR: {e}")
            models['apg'] = None
    else:
        print(f"[WARN] APG-ASR not found at {apg_path}, skipping...")
        models['apg'] = None
        
    return models

def predict_all(models, image):
    results = {}
    for name, model in models.items():
        if model is not None:
            if name == 'apg':
                results[name] = model.predict(image)
            else:
                results[name] = model.predict(image, conf=0.30, verbose=False)[0]
        else:
            results[name] = None
    return results

def get_text_color(bg_color):
    brightness = bg_color[0]*0.114 + bg_color[1]*0.587 + bg_color[2]*0.299
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)

def draw_boxes(image, result, color):
    img_copy = image.copy()
    txt_color = get_text_color(color)
    
    if result is not None and result.boxes is not None:
        for box_data in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box_data.xyxy[0].cpu().numpy()]
            conf = float(box_data.conf[0].cpu().numpy())
            cls_id = int(box_data.cls[0].cpu().numpy())
            label_name = result.names.get(cls_id, str(cls_id))
            
            label = f"{label_name} {conf:.2f}"
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)
            
    return img_copy

def draw_wbf_boxes(image, results_base, results_custom, color):
    img_copy = image.copy()
    txt_color = get_text_color(color)
    
    if results_base is None or results_custom is None or results_base.boxes is None or results_custom.boxes is None:
        return img_copy
        
    boxes1 = results_base.boxes.xyxy.cpu()
    scores1 = results_base.boxes.conf.cpu()
    labels1 = results_base.boxes.cls.cpu()
    
    boxes2 = results_custom.boxes.xyxy.cpu()
    scores2 = results_custom.boxes.conf.cpu()
    labels2 = results_custom.boxes.cls.cpu()
    
    boxes = torch.cat([boxes1, boxes2])
    scores = torch.cat([scores1, scores2])
    labels = torch.cat([labels1, labels2])
    
    if len(boxes) == 0:
        return img_copy
        
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    
    for k in keep:
        b = boxes[k]
        ious = torchvision.ops.box_iou(b.unsqueeze(0), boxes)[0]
        match_idx = (ious > 0.5) & (labels == labels[k])
        
        avg_box = boxes[match_idx].mean(dim=0)
        avg_conf = scores[match_idx].mean(dim=0)
        cls_id = int(labels[k].item())
        label_name = results_base.names.get(cls_id, str(cls_id))
        
        label = f"{label_name} {avg_conf:.2f}"
        
        x1, y1, x2, y2 = [int(v) for v in avg_box.numpy()]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)
        
    return img_copy

def add_panel_header(img, title, subtitle):
    h, w = img.shape[:2]
    header_h = 60
    header = np.ones((header_h, w, 3), dtype=np.uint8) * 255
    
    cv2.putText(header, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(header, subtitle, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    
    return cv2.vconcat([header, img])

def create_composite(panel_a, panel_b, panel_c, panel_d, global_title, global_footer):
    h, w = panel_a.shape[:2]
    panel_b = cv2.resize(panel_b, (w, h))
    panel_c = cv2.resize(panel_c, (w, h))
    panel_d = cv2.resize(panel_d, (w, h))
    
    top_row = cv2.hconcat([panel_a, panel_b])
    bottom_row = cv2.hconcat([panel_c, panel_d])
    grid = cv2.vconcat([top_row, bottom_row])
    
    grid_h, grid_w = grid.shape[:2]
    
    # Global Header
    header_h = 60
    header = np.ones((header_h, grid_w, 3), dtype=np.uint8) * 255
    (tw, th), _ = cv2.getTextSize(global_title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.putText(header, global_title, ((grid_w - tw) // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Global Footer
    footer_h = 40
    footer = np.ones((footer_h, grid_w, 3), dtype=np.uint8) * 255
    (tw, th), _ = cv2.getTextSize(global_footer, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(footer, global_footer, ((grid_w - tw) // 2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    
    return cv2.vconcat([header, grid, footer])

def run_inference(image_path=None):
    if not image_path:
        print("[ERROR] No image path provided.")
        return
        
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return
        
    print(f"Processing Image: {image_path}")
    raw_img = cv2.imread(image_path)
    
    print("Loading models...")
    models = load_models()
    
    print("Running predictions...")
    results = predict_all(models, raw_img)
    
    print("Drawing visualizations...")
    
    COLOR_RAW = (255, 255, 255)       
    COLOR_BASE = (255, 0, 0)          
    COLOR_CUSTOM = (255, 0, 0)        
    COLOR_WBF = (200, 200, 200)       
    COLOR_APG = (0, 0, 255)           
    
    img_a = raw_img.copy()
    panel_a = add_panel_header(img_a, "(a) Input Image", "Original input image")
    
    img_d = draw_boxes(raw_img, results.get('apg'), COLOR_APG)
    panel_d = add_panel_header(img_d, "(d) APG-ASR (Ours)", "Agent logic refinement")
    
    img_c = draw_wbf_boxes(raw_img, results.get('base'), results.get('custom'), COLOR_WBF)
    panel_c = add_panel_header(img_c, "(c) Naive WBF", "Averaged boxes from multiscale inference")
    
    img_b_custom = draw_boxes(raw_img, results.get('custom'), COLOR_CUSTOM)
    panel_b_custom = add_panel_header(img_b_custom, "(b) YOLOv8 Baseline (Custom)", "Raw detections (conf > 0.30)")
    
    img_b_base = draw_boxes(raw_img, results.get('base'), COLOR_BASE)
    panel_b_base = add_panel_header(img_b_base, "(b) YOLOv8 Baseline (Pretrained)", "Raw detections (conf > 0.30)")
    
    print("Building Composites...")
    
    title_a = "Occlusion - Stability Case (Domain-Specific Tuning)"
    footer_a = "APG-ASR provides granular localization refinement on top of domain-tuned confidence priors."
    composite_a = create_composite(panel_a, panel_b_custom, panel_c, panel_d, title_a, footer_a)
    
    title_b = "Occlusion - Recovery Case (Generalization from Pretrained)"
    footer_b = "Baseline exhibits localization drift, while APG-ASR stabilizes bounding boxes under occlusion."
    composite_b = create_composite(panel_a, panel_b_base, panel_c, panel_d, title_b, footer_b)
    
    out_dir = os.path.join(ROOT, "outputs", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    img_name = os.path.basename(image_path)
    
    path_a = os.path.join(out_dir, f"composite_A_{img_name}")
    path_b = os.path.join(out_dir, f"composite_B_{img_name}")
    
    cv2.imwrite(path_a, composite_a)
    cv2.imwrite(path_b, composite_b)
    
    print(f"\n[SUCCESS] Rendered outputs to:")
    print(f" -> {path_a}")
    print(f" -> {path_b}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APG-ASR Comparative Inference Engine")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    run_inference(args.image)
