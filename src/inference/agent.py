import numpy as np
import torchvision
import torch
import time

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    interArea = max(0, x2 - x1) * max(0, y2 - y1)
    if interArea == 0: return 0.0
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return float(interArea) / float(box1Area + box2Area - interArea)

class AgentLogic:
    def __init__(self, detector):
        self.detector = detector
        self.names = detector.names
        self.historical_efficiency = 1.0 
        self.total_reprocessed = 0
        self.total_effective = 0

    def get_crop(self, image, box, scale=1.0, shift_x=0.0, shift_y=0.0):
        h_img, w_img, _ = image.shape
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
        
        cx += shift_x * bw
        cy += shift_y * bh
        
        new_bw = bw * scale
        new_bh = bh * scale
        
        nx1 = max(0, int(cx - new_bw / 2.0))
        ny1 = max(0, int(cy - new_bh / 2.0))
        nx2 = min(w_img, int(cx + new_bw / 2.0))
        ny2 = min(h_img, int(cy + new_bh / 2.0))
        
        if nx2 <= nx1 or ny2 <= ny1:
            return None, None
            
        crop = image[ny1:ny2, nx1:nx2]
        return crop, (nx1, ny1)

    def run_agent(self, image, raw_detections, max_reprocess=5, telemetry=None):
        stats = {
            'strong_trust': 0, 'weak_trust': 0, 'discarded': 0,
            'reprocessed': 0, 'upgraded': 0, 'rejected': 0,
            'context_boosts': 0, 'effective_reprocess': 0, 'wasted_reprocess': 0,
            'failures': {
                'crop_failed': 0, 'lost_detection': 0, 'reprocess_conf_drop': 0,
                'suppressed_by_nms': 0, 'class_skip': 0, 'timeout_skip': 0
            }
        }
        
        t0 = time.time()
        final_detections = []
        uncertain_candidates = []
        
        scene_objects = {}
        for det in raw_detections:
            cls_name = self.names.get(det['cls'], str(det['cls'])).lower()
            scene_objects[cls_name] = scene_objects.get(cls_name, 0) + 1
            
        many_persons = scene_objects.get('person', 0) >= 3
        
        for det in raw_detections:
            conf = det['conf']
            cls = det['cls']
            cls_name = self.names.get(cls, str(cls)).lower()
            
            if many_persons and cls_name == "dining table" and 0.35 <= conf < 0.6:
                det['conf'] = min(conf * 1.3, 0.9)
                det['source'] = 'context_boosted'
                final_detections.append(det)
                stats['context_boosts'] += 1
                stats['strong_trust'] += 1
                continue
            
            if conf >= 0.6:
                det['source'] = 'strong_trust'
                final_detections.append(det)
                stats['strong_trust'] += 1
            elif 0.15 <= conf < 0.6:
                if conf >= 0.4:
                    det['source'] = 'weak_trust'
                    stats['weak_trust'] += 1
                else:
                    det['source'] = 'uncertain'
                
                if cls_name == "person" and conf < 0.25:
                    stats['failures']['class_skip'] += 1
                    stats['discarded'] += 1
                    continue
                    
                x1, y1, x2, y2 = [int(v) for v in det['coords']]
                area = max((x2 - x1) * (y2 - y1), 1)
                
                uncertainty = (0.5 - conf) * 2.0
                size_weight = min((10000.0 / area), 5.0)
                class_weight = 1.0
                if cls_name in ["bottle", "tie", "cup", "wine glass", "fork"]:
                    class_weight = 2.0
                    
                det['priority_score'] = uncertainty + size_weight + class_weight
                uncertain_candidates.append(det)
            else:
                stats['discarded'] += 1
                
        uncertain_candidates.sort(key=lambda x: (-x['priority_score'], x['cls']))
        adaptive_k = max(2, int(len(uncertain_candidates) * 0.5))
        if self.historical_efficiency < 0.2 and adaptive_k > 1:
            adaptive_k = max(1, adaptive_k - 1)
        adaptive_k = min(adaptive_k, 5)
        
        skipped_candidates = uncertain_candidates[adaptive_k:]
        for sc in skipped_candidates:
            if sc['conf'] >= 0.3:
                sc['source'] = 'retained'
                final_detections.append(sc)
                
        uncertain_candidates = uncertain_candidates[:adaptive_k]
        
        print(f"\n[STAGE 2] Partitioning detections...")
        print(f"Strong trust (>=0.6): {stats['strong_trust']}")
        print(f"Weak trust (0.4-0.6): {stats['weak_trust']}")
        print(f"Context boosts applied: {stats['context_boosts']}")
        print(f"Discarded (<0.15 or class-skip): {stats['discarded']}")
        print(f"Adaptive K limits processing to {adaptive_k} candidates.")
        
        agent_trace = []
        
        # Perturbation configs: (scale, shift_x, shift_y)
        perturbations = [
            (1.6, 0.0, 0.0),
            (1.2, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.5, 0.0, 0.0),
            (1.6, 0.2, 0.0),
            (1.6, -0.2, 0.0),
            (1.6, 0.0, 0.2),
            (1.6, 0.0, -0.2)
        ]
        
        print(f"\n[STAGE 3] Reprocessing uncertain detections...")
        for i, candidate in enumerate(uncertain_candidates):
            orig_conf = candidate['conf']
            orig_coords = candidate['coords']
            orig_cls = candidate['cls']
            obj_id = candidate.get('id', f"obj_unk_{i}")
            cls_name = self.names.get(orig_cls, str(orig_cls))
            
            trace_entry = {
                "object_id": obj_id,
                "class": cls_name,
                "original_conf": float(orig_conf),
                "reprocessed_conf": 0.0,
                "decision": "skipped",
                "reason": ""
            }
            
            print(f"\n-> Candidate {i+1}: {cls_name} (conf={orig_conf:.2f})")
            
            all_candidates = []
            
            for p_scale, p_sx, p_sy in perturbations:
                crop_img, offset = self.get_crop(image, orig_coords, scale=p_scale, shift_x=p_sx, shift_y=p_sy)
                if crop_img is None or crop_img.size == 0:
                    continue
                    
                crop_dets = self.detector.run_on_crop(crop_img)
                stats['reprocessed'] += 1
                
                valid_dets = []
                for cdet in crop_dets:
                    if cdet['cls'] == orig_cls:
                        cx1, cy1, cx2, cy2 = cdet['coords']
                        ox, oy = offset
                        global_coords = np.array([cx1 + ox, cy1 + oy, cx2 + ox, cy2 + oy])
                        valid_dets.append({'coords': global_coords, 'conf': cdet['conf']})
                        
                if valid_dets:
                    best_c = max(valid_dets, key=lambda x: x['conf'])
                    all_candidates.append(best_c)
                    
            if not all_candidates:
                stats['failures']['lost_detection'] += 1
                stats['wasted_reprocess'] += 1
                print(f"   [FAIL] lost_detection (no boxes in any perturbation)")
                print(f"   -> RESULT: REJECTED (Fallback to original)")
                trace_entry["decision"] = "rejected"
                trace_entry["reason"] = "lost_detection"
                agent_trace.append(trace_entry)
                
                candidate['source'] = 'retained_fallback'
                final_detections.append(candidate)
                stats['rejected'] += 1
                continue
                
            # Pairwise IoU Check (Variance Trigger)
            pairwise_ious = []
            n_cands = len(all_candidates)
            if n_cands > 1:
                for a in range(n_cands):
                    for b in range(a+1, n_cands):
                        pairwise_ious.append(compute_iou(all_candidates[a]['coords'], all_candidates[b]['coords']))
                mean_iou = float(np.mean(pairwise_ious))
            else:
                mean_iou = 1.0
                
            if mean_iou > 0.6:
                # Stable box
                stats['rejected'] += 1
                print(f"   [INFO] Stable prediction (mean_iou={mean_iou:.2f} > 0.6)")
                print(f"   -> RESULT: STABLE (Fallback to original)")
                trace_entry["decision"] = "stable_skip"
                trace_entry["reason"] = "high_agreement"
                agent_trace.append(trace_entry)
                
                candidate['source'] = 'retained_stable'
                final_detections.append(candidate)
                continue
                
            # Unstable -> Consistency Weighted Aggregation
            for c in all_candidates:
                ious = [compute_iou(c['coords'], other['coords']) for other in all_candidates if other is not c]
                c['consistency'] = float(np.mean(ious)) if ious else 1.0
                c['score'] = c['conf'] * c['consistency']
                
            best_box = max(all_candidates, key=lambda x: x['score'])
            max_consistency = best_box['consistency']
            
            # Recall Protection (Fallback)
            if max_consistency < 0.3:
                stats['rejected'] += 1
                print(f"   [WARN] Unstable garbage (max_consistency={max_consistency:.2f} < 0.3)")
                print(f"   -> RESULT: FALLBACK (Recall protected)")
                trace_entry["decision"] = "fallback"
                trace_entry["reason"] = "low_consistency"
                agent_trace.append(trace_entry)
                
                candidate['source'] = 'retained_fallback'
                final_detections.append(candidate)
                continue
                
            # Boundary-Aware Expansion
            final_coords = best_box['coords'].copy()
            if max_consistency < 0.99 and best_box['conf'] > 0.1:
                bw = final_coords[2] - final_coords[0]
                bh = final_coords[3] - final_coords[1]
                cx = final_coords[0] + bw / 2.0
                cy = final_coords[1] + bh / 2.0
                nbw = bw * 1.05
                nbh = bh * 1.05
                final_coords = np.array([
                    max(0, cx - nbw / 2.0),
                    max(0, cy - nbh / 2.0),
                    min(image.shape[1], cx + nbw / 2.0),
                    min(image.shape[0], cy + nbh / 2.0)
                ])
                print(f"   [INFO] Expanded boundary (consistency={max_consistency:.2f})")
                
            # Final Success
            print(f"   [SUCCESS] Relocalized (consistency={max_consistency:.2f})")
            print(f"   -> RESULT: UPGRADED")
            
            trace_entry["reprocessed_conf"] = float(best_box['conf'])
            trace_entry["decision"] = "upgraded"
            trace_entry["reason"] = "consistency_selection"
            agent_trace.append(trace_entry)
            
            best_new_det = {
                'coords': final_coords,
                'conf': float(orig_conf),  # Confidence locking
                'old_conf': float(orig_conf),
                'cls': orig_cls,
                'source': 'reprocessed_upgrade',
                'id': obj_id
            }
            final_detections.append(best_new_det)
            stats['upgraded'] += 1
            stats['effective_reprocess'] += 1
            
        self.total_reprocessed += stats['reprocessed']
        self.total_effective += stats['effective_reprocess']
        if self.total_reprocessed > 0:
            self.historical_efficiency = self.total_effective / self.total_reprocessed
                
        t1 = time.time()
        
        t2 = time.time()
        if len(final_detections) > 0:
            boxes = torch.tensor(np.array([d['coords'] for d in final_detections]), dtype=torch.float32)
            scores = torch.tensor([d['conf'] for d in final_detections], dtype=torch.float32)
            
            pre_nms_len = len(final_detections)
            keep_indices = torchvision.ops.nms(boxes, scores, 0.45)
            post_nms_len = len(keep_indices)
            
            stats['failures']['suppressed_by_nms'] = pre_nms_len - post_nms_len
            final_detections = [final_detections[i] for i in keep_indices.numpy()]
            
            final_detections = [d for d in final_detections if d['conf'] >= 0.3 or d['source'] == 'context_boosted']
            
        t3 = time.time()
        
        timings = {
            'agent_time_ms': (t1 - t0) * 1000,
            'nms_time_ms': (t3 - t2) * 1000
        }
            
        structured_output = []
        for d in final_detections:
            cls_id = int(d['cls'])
            structured_output.append({
                "id": d.get('id', f"obj_new_{cls_id}_{time.time()}"),
                "class_id": cls_id,
                "class": self.names.get(cls_id, str(cls_id)),
                "confidence": float(d.get('old_conf', d['conf'])),
                "source": d['source'],
                "bbox": [float(v) for v in d['coords']]
            })
            
        return structured_output, stats, timings, agent_trace




