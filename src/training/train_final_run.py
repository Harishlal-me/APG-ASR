import sys
import os
import shutil
import torch
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.training.train_engine import APG_ASRTrainer

def cleanup_dirs():
    for d in ["yolo_final_run", "yolo_internal"]:
        path = os.path.join(ROOT, d)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Cleaned up {path}")
    # Also clean telemetry.log
    tel_log = os.path.join(ROOT, "telemetry.log")
    if os.path.exists(tel_log):
        os.remove(tel_log)
        print("Cleaned up telemetry.log")

def run_test():
    cleanup_dirs()
    
    data_path = os.path.join(ROOT, "config", "dataset.yaml")
    
    # Strictly start from Epoch 80 baseline (0.319)
    resume_path = os.path.join(ROOT, "phase_results", "Phase_8", "weights", "epoch80.pt")
    if not os.path.exists(resume_path):
        print(f"[ERROR] Cannot find {resume_path}")
        sys.exit(1)
        
    os.environ["APG_RESUME_WEIGHTS"] = resume_path
    
    # FINAL RUN: GATE = 0.3
    os.environ["APG_GATE_MULTIPLIER"] = "0.3"
    
    # UNFREEZE TRANSFORMER (Phase 4 mode)
    os.environ["APG_PHASE"] = "4"

    overrides = {
        "model": "yolov8n.yaml",
        "data": data_path,
        "epochs": 20,          # 15-20 continuous epochs
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
        "lr0": 3e-5,          # STRICT fixed LR
        "lrf": 1.0,           
        "weight_decay": 0.0005,
        "cos_lr": False,      # NO cosine restart
        "warmup_epochs": 0,   # NO warmup
        "val": True,
        "save": True,
        "save_period": 1,
        "project": ROOT,
        "name": "yolo_final_run",
        "exist_ok": True,
        "seed": 42,
    }

    trainer = APG_ASRTrainer(overrides=overrides)
    trainer.train()

if __name__ == "__main__":
    run_test()

