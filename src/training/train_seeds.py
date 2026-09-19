import sys
import os
import shutil
import torch
import gc
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.training.train_engine import APG_ASRTrainer

def run_seed_test(seed):
    name = f"yolo_final_run_seed{seed}"
    print(f"\n{'='*50}")
    print(f"ðŸš€ STARTING MULTI-SEED VALIDATION: SEED {seed}")
    print(f"{'='*50}\n")
    
    data_path = os.path.join(ROOT, "config", "dataset.yaml")
    
    resume_path = os.path.join(ROOT, "phase_results", "Phase_8", "weights", "epoch80.pt")
    if not os.path.exists(resume_path):
        print(f"[ERROR] Cannot find {resume_path}")
        sys.exit(1)
        
    os.environ["APG_RESUME_WEIGHTS"] = resume_path
    os.environ["APG_GATE_MULTIPLIER"] = "0.3"
    os.environ["APG_PHASE"] = "4"
    os.environ["APG_TELEMETRY_FILE"] = f"telemetry_seed{seed}.log"

    overrides = {
        "model": "yolov8n.yaml",
        "data": data_path,
        "epochs": 20,          
        "batch": 4,
        "imgsz": 640,
        "device": 0,
        "amp": True,
        "workers": 2,
        "cache": False,
        "mosaic": 1.0,
        "mixup": 0.0,
        "close_mosaic": 2,
        "optimizer": "AdamW",
        "lr0": 3e-5,          
        "lrf": 1.0,           
        "weight_decay": 0.0005,
        "cos_lr": False,      
        "warmup_epochs": 0,   
        "val": True,
        "save": True,
        "save_period": 1,
        "project": ROOT,
        "name": name,
        "exist_ok": True,
        "seed": seed,
    }

    trainer = APG_ASRTrainer(overrides=overrides)
    trainer.train()

if __name__ == "__main__":
    seeds_to_test = [999]  # Final run for Seed 999
    for seed in seeds_to_test:
        run_seed_test(seed)
        
        # Clear CUDA memory between heavy sequential runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    print("\nâœ… ALL SEED RUNS COMPLETED SUCCESSFULLY.")

