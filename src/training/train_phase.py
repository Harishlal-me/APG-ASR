"""
APG-ASR Unified Phase Runner

Usage:
    python scripts/train_phase.py --phase 1                 # Phase 1: epochs 1-10
    python scripts/train_phase.py --phase 2                 # Phase 2: epochs 11-20 (resumes)
    python scripts/train_phase.py --phase 1 --test-epochs 1 # Integration test: 1 epoch only

Phase schedule (10 phases Ã— 10 epochs = 100 total):
    Phase 1:  epochs  1-10
    Phase 2:  epochs 11-20
    Phase 3:  epochs 21-30
    Phase 4:  epochs 31-40
    Phase 5:  epochs 41-50
    Phase 6:  epochs 51-60
    Phase 7:  epochs 61-70
    Phase 8:  epochs 71-80
    Phase 9:  epochs 81-90
    Phase 10: epochs 91-100
"""

import os
import sys
import time
import csv
import shutil
import argparse
import torch
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.training.train_engine import train_phase

PROJECT_ROOT = Path(ROOT)
YOLO_RUN_DIR = PROJECT_ROOT / "yolo_internal"
EPOCHS_PER_PHASE = 10


def check_gpu():
    """Verify GPU is available and print hardware info."""
    print("\n=======================================================")
    print("[INFO] Verifying Hardware & GPU Engagement...")
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is NOT available. Stopping now.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"[SUCCESS] GPU Detected: {device_name}")
    print(f"[INFO]    VRAM: {vram_mb:.0f} MB")
    print("=======================================================\n")


def sync_outputs(phase, target_epoch):
    """Copy outputs from yolo_internal/ to a structured phase directory."""
    print(f"[INFO] Synchronizing Phase {phase} outputs...")

    weights_dir = YOLO_RUN_DIR / "weights"
    
    # Create the phase-specific directory structure
    phase_dir = PROJECT_ROOT / "phase_results" / f"Phase_{phase}"
    phase_weights_dir = phase_dir / "weights"
    phase_plots_dir = phase_dir / "plots"
    phase_logs_dir = phase_dir / "logs"
    
    for d in (phase_weights_dir, phase_plots_dir, phase_logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Copy checkpoints
    if (weights_dir / "last.pt").exists():
        shutil.copy(weights_dir / "last.pt", phase_weights_dir / "last.pt")
        shutil.copy(weights_dir / "last.pt", phase_weights_dir / f"epoch{target_epoch}.pt")
        print(f"[INFO] Saved {phase_weights_dir}/epoch{target_epoch}.pt")
    if (weights_dir / "best.pt").exists():
        shutil.copy(weights_dir / "best.pt", phase_weights_dir / "best.pt")

    # Copy results CSV
    if (YOLO_RUN_DIR / "results.csv").exists():
        shutil.copy(YOLO_RUN_DIR / "results.csv", phase_logs_dir / "results.csv")

    # Copy plots
    for ext in ("*.png", "*.jpg"):
        for plot in YOLO_RUN_DIR.glob(ext):
            shutil.copy(plot, phase_plots_dir / plot.name)

    # Copy config
    if (YOLO_RUN_DIR / "args.yaml").exists():
        shutil.copy(YOLO_RUN_DIR / "args.yaml", phase_dir / "args.yaml")

    # Final model (only on last phase)
    if phase == 10:
        final_dir = PROJECT_ROOT / "final_model"
        final_dir.mkdir(exist_ok=True)
        if (weights_dir / "best.pt").exists():
            shutil.copy(weights_dir / "best.pt", final_dir / "apg_asr_best.pt")
            print("[INFO] Final best model saved to final_model/apg_asr_best.pt")

    # Write phase note
    start_epoch = (phase - 1) * EPOCHS_PER_PHASE + 1
    note_path = phase_dir / "note.md"
    note_content = f"""# Phase {phase}
Epochs: {start_epoch}â€“{target_epoch}
Batch Size: 4
Image Size: 640
GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
Precision: FP16 (AMP)
Optimizer: AdamW (lr=0.001, wd=0.0005)
Notes: APG-ASR Phase {phase} training completed successfully.
"""
    note_path.write_text(note_content)
    print(f"[INFO] Phase {phase} results successfully saved to phase_results/Phase_{phase}/")

    # Clean up the working directory so it doesn't trigger a false resume on the next phase
    if YOLO_RUN_DIR.exists():
        shutil.rmtree(YOLO_RUN_DIR)
        print("[INFO] Cleaned up yolo_internal working directory.")


def main():
    parser = argparse.ArgumentParser(description="APG-ASR Phase Training Runner")
    parser.add_argument("--phase", type=int, required=True, choices=range(1, 11),
                        help="Phase number (1-10)")
    parser.add_argument("--test-epochs", type=int, default=None,
                        help="Override: run exactly this many epochs (for integration testing)")
    args = parser.parse_args()

    phase_start_time = time.time()

    phase = args.phase
    os.environ["APG_PHASE"] = str(phase)
    start_epoch = (phase - 1) * EPOCHS_PER_PHASE + 1
    target_epoch = phase * EPOCHS_PER_PHASE

    # If --test-epochs is given, override the target
    if args.test_epochs is not None:
        target_epoch = (phase - 1) * EPOCHS_PER_PHASE + args.test_epochs
        print(f"[TEST MODE] Running {args.test_epochs} epoch(s) only")

    print(f"{'=' * 60}")
    print(f"  APG-ASR Phase {phase} Training")
    print(f"  Epochs: {start_epoch} â†’ {target_epoch}")
    print(f"{'=' * 60}")

    check_gpu()

    # Determine resume path
    resume_path = None
    
    # 1. Always check if there's an active run in yolo_internal we can resume from
    internal_ckpt = str(YOLO_RUN_DIR / "weights" / "last.pt")
    
    # 2. Check the structured phase_results directory for the CURRENT phase (if it crashed mid-phase)
    curr_phase_ckpt = str(PROJECT_ROOT / "phase_results" / f"Phase_{phase}" / "weights" / "last.pt")
    
    # 3. Check the structured phase_results directory for the PREVIOUS phase
    prev_phase_ckpt = str(PROJECT_ROOT / "phase_results" / f"Phase_{phase-1}" / "weights" / "last.pt") if phase > 1 else None
    
    is_true_resume = False
    if os.path.exists(internal_ckpt):
        resume_path = internal_ckpt
        is_true_resume = True
    elif os.path.exists(curr_phase_ckpt):
        resume_path = curr_phase_ckpt
        is_true_resume = True
    elif prev_phase_ckpt and os.path.exists(prev_phase_ckpt):
        # We cannot use resume=True across phases because Ultralytics strips the optimizer state 
        # from the checkpoint when a phase completes.
        resume_path = prev_phase_ckpt
        is_true_resume = False
    elif phase > 1:
        print(f"[ERROR] Cannot resume Phase {phase}. Checkpoint not found.")
        print(f"        Expected: {prev_phase_ckpt}")
        print(f"        Run Phase {phase - 1} first.")
        sys.exit(1)
        
    if resume_path:
        if is_true_resume:
            print(f"[INFO] Resuming mid-phase from: {resume_path}")
        else:
            print(f"[INFO] Initializing new phase from previous weights: {resume_path}")

    # Run training
    # total_epochs = target_epoch for true resume (Ultralytics handles start_epoch)
    # If not a true resume, we just want to train for EPOCHS_PER_PHASE epochs
    run_epochs = target_epoch if is_true_resume else EPOCHS_PER_PHASE
    train_phase(total_epochs=run_epochs, resume_path=resume_path, is_true_resume=is_true_resume, phase=phase)

    # Sync outputs
    sync_outputs(phase, target_epoch)

    phase_end_time = time.time()
    duration_sec = int(phase_end_time - phase_start_time)
    hours, remainder = divmod(duration_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Log to phase_times.csv
    times_csv_path = PROJECT_ROOT / "phase_results" / "phase_times.csv"
    file_exists = times_csv_path.exists()
    with open(times_csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Phase", "Start_Epoch", "End_Epoch", "Duration_Seconds", "Duration_Formatted"])
        writer.writerow([phase, start_epoch, target_epoch, duration_sec, formatted_time])

    print(f"\n{'=' * 60}")
    print(f"  Phase {phase} Complete â€” epoch{target_epoch}.pt saved")
    print(f"  Time taken: {formatted_time}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

