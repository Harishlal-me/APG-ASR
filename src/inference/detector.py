import cv2
from ultralytics import YOLO

class BaseDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        self.names = self.model.names if hasattr(self.model, 'names') else {}

    def get_raw_detections(self, image, conf=0.1):
        results = self.model.predict(image, conf=conf, verbose=False)[0]
        raw_detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box_data in results.boxes:
                coords = box_data.xyxy[0].cpu().numpy()
                score = float(box_data.conf[0].cpu().numpy())
                cls = int(box_data.cls[0].cpu().numpy())
                raw_detections.append({'coords': coords, 'conf': score, 'cls': cls})
        return raw_detections

    def run_on_crop(self, crop_img):
        results = self.model.predict(crop_img, conf=0.1, verbose=False)[0]
        dets = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box_data in results.boxes:
                coords = box_data.xyxy[0].cpu().numpy()
                score = float(box_data.conf[0].cpu().numpy())
                cls = int(box_data.cls[0].cpu().numpy())
                dets.append({'coords': coords, 'conf': score, 'cls': cls})
        return dets

    def get_apg_telemetry(self):
        try:
            gate4 = self.model.model.apg.gate.gate4
            gate5 = self.model.model.apg.gate.gate5
            
            def safe_get(g):
                a = getattr(g, 'alpha_mean_ema', 0.0)
                t = getattr(g, 't_norm_ema', 0.0)
                p = getattr(g, 'p_norm_ema', 1e-6)
                if p == 0: p = 1e-6
                return a, t, p
                
            a4, t4, p4 = safe_get(gate4)
            a5, t5, p5 = safe_get(gate5)
            
            r4 = (a4 * t4) / p4
            r5 = (a5 * t5) / p5
            
            return {
                'P4': {'alpha': a4, 't_norm': t4, 'p_norm': p4, 'ratio': r4},
                'P5': {'alpha': a5, 't_norm': t5, 'p_norm': p5, 'ratio': r5}
            }
        except Exception as e:
            return None
