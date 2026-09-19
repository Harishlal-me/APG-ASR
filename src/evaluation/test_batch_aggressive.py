import os
import sys
import glob

# Ensure run_pipeline is accessible
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from experiments.apg_asr.scripts.run_pipeline import run_image_pipeline

def main():
    image_dir = r"d:\cdacminor\datasets\coco_original\val2017"
    all_images = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    # We will test on exactly 100 images to get a statistically significant batch
    images_to_test = all_images[:100]
    
    out_md = r"d:\cdacminor\experiments\apg_asr\batch_results_aggressive.md"
    
    global_metrics = {
        "latency_sum": 0.0,
        "efficiency_sum": 0.0,
        "total_reprocessed": 0,
        "total_upgraded": 0,
        "failures": {}
    }
    
    all_json_results = []
    latencies = []
    
    with open(out_md, "w") as f:
        f.write("# APG-ASR Agentic Layer Batch Evaluation (AGGRESSIVE MODE)\n\n")
        
        for i, img_path in enumerate(images_to_test):
            img_name = os.path.basename(img_path)
            print(f"[{i+1}/100] Processing {img_name}...")
            
            try:
                res = run_image_pipeline(img_path)
                
                # --- NEW: SAVE IMAGES ---
                import shutil
                import cv2
                
                raw_dir = r"d:\cdacminor\experiments\apg_asr\batch_outputs\raw"
                proc_dir = r"d:\cdacminor\experiments\apg_asr\batch_outputs\processed_aggressive"
                os.makedirs(raw_dir, exist_ok=True)
                os.makedirs(proc_dir, exist_ok=True)
                
                # Copy raw
                shutil.copy2(img_path, os.path.join(raw_dir, img_name))
                
                # Draw and save processed
                out_img = cv2.imread(img_path)
                for d in res["final_detections"]:
                    x1, y1, x2, y2 = [int(v) for v in d['bbox']]
                    if d['source'] == 'strong_trust':
                        color = (0, 255, 0)
                        label = f"{d['class']} {d['confidence']:.2f}"
                    elif d['source'] == 'context_boosted':
                        color = (0, 255, 255)
                        label = f"{d['class']} {d['confidence']:.2f} (boost)"
                    elif d['source'] == 'reprocessed_upgrade':
                        color = (255, 0, 0)
                        old_c = d.get('old_conf', 0.0)
                        label = f"{d['class']} {old_c:.2f}->{d['confidence']:.2f} (UP)"
                    else:
                        color = (0, 165, 255)
                        label = f"{d['class']} {d['confidence']:.2f}"
                        
                    cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(out_img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                    cv2.putText(out_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                cv2.imwrite(os.path.join(proc_dir, img_name), out_img)
                
                # Update global metrics
                latencies.append(res["latency_ms"])
                global_metrics["latency_sum"] += res["latency_ms"]
                global_metrics["efficiency_sum"] += res["efficiency"]
                global_metrics["total_reprocessed"] += res["reprocessed"]
                global_metrics["total_upgraded"] += res["upgraded"]
                
                for fail_type, count in res["failures"].items():
                    global_metrics["failures"][fail_type] = global_metrics["failures"].get(fail_type, 0) + count
                    
                # Store JSON record
                record = {
                    "image": img_name,
                    "raw_count": len(res["raw_detections"]),
                    "final_count": len(res["final_detections"]),
                    "reprocessed": res["reprocessed"],
                    "upgraded": res["upgraded"],
                    "rejected": res["reprocessed"] - res["upgraded"],
                    "efficiency": res["efficiency"],
                    "latency_ms": res["latency_ms"],
                    "failures": res["failures"],
                    "agent_trace": res["agent_trace"]
                }
                all_json_results.append(record)
                
                # Write to markdown
                f.write(f"## Image: {img_name}\n\n")
                
                f.write("### Raw Detections\n")
                if len(res["raw_detections"]) == 0:
                    f.write("- None\n")
                for d in res["raw_detections"]:
                    f.write(f"- {d['class']} ({d['conf']:.2f})\n")
                    
                f.write("\n### Agent Decisions\n")
                if len(res["agent_trace"]) == 0:
                    f.write("- None (no candidates processed)\n")
                for trace in res["agent_trace"]:
                    decision = trace['decision'].upper()
                    reason = f" ({trace['reason']})" if trace['reason'] else ""
                    f.write(f"- {trace['class']}: {trace['original_conf']:.2f} -> {trace['reprocessed_conf']:.2f} -> {decision}{reason}\n")
                    
                f.write("\n### Final Detections\n")
                if len(res["final_detections"]) == 0:
                    f.write("- None\n")
                for d in res["final_detections"]:
                    f.write(f"- {d['class']} ({d['confidence']:.2f}) [{d['source']}]\n")
                    
                f.write("\n### Metrics\n")
                f.write(f"- Latency: {res['latency_ms']:.2f} ms\n")
                f.write(f"- Efficiency: {res['efficiency']:.1f}%\n")
                f.write(f"- Reprocessed: {res['reprocessed']}\n")
                f.write(f"- Upgraded: {res['upgraded']}\n")
                f.write("\n---\n\n")
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                f.write(f"## Image: {img_name}\n\n- ERROR: {e}\n\n---\n\n")

    import json
    json_out = r"d:\cdacminor\experiments\apg_asr\batch_results_aggressive.json"
    with open(json_out, "w") as jf:
        json.dump(all_json_results, jf, indent=2)

    # Final summary calculations
    avg_latency = global_metrics["latency_sum"] / 50
    avg_eff = global_metrics["efficiency_sum"] / 50
    
    print("\n\n=========================================")
    print("         BATCH TESTING COMPLETE")
    print("=========================================\n")
    print(f"Latency Distribution:")
    print(f"  - Avg: {avg_latency:.2f} ms")
    print(f"  - Min: {min(latencies):.2f} ms")
    print(f"  - Max: {max(latencies):.2f} ms")
    
    print(f"\nAverage Efficiency: {avg_eff:.1f}%")
    print(f"Total Reprocessed: {global_metrics['total_reprocessed']}")
    print(f"Total Upgraded: {global_metrics['total_upgraded']}")
    
    print("\nFailure Distribution:")
    total_failures = sum(global_metrics["failures"].values())
    for ftype, count in global_metrics["failures"].items():
        pct = (count / max(1, total_failures)) * 100
        print(f"  - {ftype}: {count} ({pct:.1f}%)")
        
    print(f"\nDetailed markdown log written to: {out_md}")
    print(f"Detailed JSON results written to: {json_out}")

if __name__ == "__main__":
    main()

