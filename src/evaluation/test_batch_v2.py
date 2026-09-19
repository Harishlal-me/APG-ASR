import os
import sys
import glob
import cv2
import time
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from experiments.apg_asr.scripts.comprehensive_eval import TelemetryAgent

def main():
    image_dir = r"d:\cdacminor\datasets\coco_50k\val"
    all_images = glob.glob(os.path.join(image_dir, "*.jpg"))
    images_to_test = all_images[:100]
    
    weights_path = r"d:\cdacminor\experiments\apg_asr\yolo_final_run_seed999\weights\best.pt"
    
    # Initialize our newest agent with the STRICT confidence gate
    agent = TelemetryAgent(weights_path, tau_low=0.10, tau_high=0.30, top_k=15)
    class_names = agent.model.names
    
    raw_dir = r"d:\cdacminor\experiments\apg_asr\batch_outputs\afterexperiments\raw"
    proc_dir = r"d:\cdacminor\experiments\apg_asr\batch_outputs\afterexperiments\processed"
    final_dir = r"d:\cdacminor\experiments\apg_asr\batch_outputs\afterexperiments\finalprocessed"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    print(f"Starting batch visualization on {len(images_to_test)} images...")
    
    total_upgrades = 0
    
    for i, img_path in enumerate(images_to_test):
        img_name = os.path.basename(img_path)
        print(f"[{i+1}/100] Processing {img_name}...")
        
        # 1. RAW (Pure original image with no boxes)
        shutil.copy2(img_path, os.path.join(raw_dir, img_name))
        
        # 2. PROCESSED & FINAL PROCESSED (Agentic Predictions)
        # run_telemetry_inference uses 0.001 internal conf, so we filter down to 0.10 for viz
        final_boxes, final_scores, final_classes, telemetry, lat = agent.run_telemetry_inference(img_path)
        
        proc_img = cv2.imread(img_path)
        final_img = cv2.imread(img_path)
        
        # Build a list of upgraded coordinates to highlight them
        upgraded_coords = []
        for t in telemetry:
            if t['status'] == 'upgraded':
                upgraded_coords.append(t['new_coords'])
                total_upgrades += 1
                
        for idx in range(len(final_boxes)):
            conf = final_scores[idx]
            if conf < 0.10: continue # Skip viz for very low conf
            
            x1, y1, x2, y2 = final_boxes[idx]
            cls = int(final_classes[idx])
            cls_name = class_names.get(cls, str(cls))
            
            # --- Draw on PROCESSED (Standard Blue Boxes) ---
            proc_color = (255, 0, 0)
            proc_label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(proc_img, (int(x1), int(y1)), (int(x2), int(y2)), proc_color, 2)
            cv2.putText(proc_img, proc_label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, proc_color, 1)
            
            # --- Draw on FINAL PROCESSED (Color Coded Agent Behavior) ---
            # Check if this specific box was upgraded
            is_upgraded = False
            for uc in upgraded_coords:
                if list(uc) == list(final_boxes[idx]):
                    is_upgraded = True
                    break
                    
            if is_upgraded:
                # Highlight in RED
                f_color = (0, 0, 255)
                f_label = f"{cls_name} {conf:.2f} (UPGRADED)"
                thick = 3
            elif conf >= 0.30:
                # Highlight in GREEN (Frozen / Untouched)
                f_color = (0, 255, 0)
                f_label = f"{cls_name} {conf:.2f} (FROZEN)"
                thick = 2
            else:
                # Highlight in BLUE (Rejected by Agent)
                f_color = (255, 0, 0)
                f_label = f"{cls_name} {conf:.2f} (RAW)"
                thick = 1
                
            cv2.rectangle(final_img, (int(x1), int(y1)), (int(x2), int(y2)), f_color, thick)
            cv2.putText(final_img, f_label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, f_color, 1)
            
        cv2.imwrite(os.path.join(proc_dir, img_name), proc_img)
        cv2.imwrite(os.path.join(final_dir, img_name), final_img)

    print("\n=========================================")
    print("      BATCH VISUALIZATION COMPLETE")
    print("=========================================")
    print(f"Total Images: {len(images_to_test)}")
    print(f"Total Upgrades Highlighted: {total_upgrades}")
    print(f"Raw Images saved to: {raw_dir}")
    print(f"Processed Images saved to: {proc_dir}")
    print(f"Final Processed Images saved to: {final_dir}")

if __name__ == "__main__":
    main()

