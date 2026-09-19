import os
import sys
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from src.training.train_engine import UltralyticsAPGWrapper
from ultralytics.models.yolo.detect.val import DetectionValidator

def main():
    best_weights = r"d:\cdacminor\experiments\apg_asr\yolo_final_run_seed999\weights\best.pt"
    data_yaml = r"d:\cdacminor\experiments\apg_asr\config\dataset.yaml"
    
    print(f"\n{'='*50}")
    print("ðŸš€ STARTING FINAL TEST SET EVALUATION")
    print(f"{'='*50}\n")
    print(f"Weights: {best_weights}")
    
    with open(data_yaml, "r") as f:
        data_dict = yaml.safe_load(f)
    nc = data_dict.get("nc", 80)
    names = data_dict.get("names", {i: str(i) for i in range(nc)})
    
    model = UltralyticsAPGWrapper(nc, names)
    
    # Manual weight loading
    ckpt = torch.load(best_weights, map_location='cpu', weights_only=False)
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
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights with {len(missing)} missing keys.")
    model.eval()

    args = dict(
        model=best_weights,
        data=data_yaml,
        split='test',
        batch=4,
        imgsz=640,
        device=0,
        name='final_test_eval_seed999',
        project=os.path.join(ROOT, "experiments", "apg_asr")
    )
    
    # Must enforce APG Gate logic in env
    os.environ["APG_GATE_MULTIPLIER"] = "0.3"

    validator = DetectionValidator(args=args)
    
    # Overwrite the validator model loading to prevent standard YOLO loading
    validator.model = model.to('cuda:0')
    
    print("\n[INFO] Launching Validation against 5K Test Set...\n")
    results = validator(model=validator.model)
    
    print("\nâœ… TEST SET EVALUATION COMPLETE")
    # Validator returns a dict of metrics (if wrapped properly), or sets attributes.
    print(results)

if __name__ == "__main__":
    main()

