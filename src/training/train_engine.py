import sys
import os
import torch
import torch.nn as nn

# Ensure the repository root is in the path for internal imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors, dist2bbox
from src.models.apg_asr.complete_model import APG_ASR


# ---------------------------------------------------------------------------
# DummyHead: Provides .nc, .reg_max, .stride, .no attributes that
# v8DetectionLoss.__init__ reads from src.models.apg_asr.model[-1].
# ---------------------------------------------------------------------------
class DummyHead:
    def __init__(self, nc, reg_max, stride):
        self.nc = nc
        self.no = nc + reg_max * 4
        self.reg_max = reg_max
        self.stride = stride

    def _get_name(self):
        return "Detect"


# ---------------------------------------------------------------------------
# DFL: Distribution Focal Loss integral (same as Ultralytics DFL).
# Converts reg_max distribution to single expected value per coordinate.
# Needed for inference-time box decoding.
# ---------------------------------------------------------------------------
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# ---------------------------------------------------------------------------
# _ModelProxy: Makes model.model[-1] return the DummyHead.
# Stored as a plain Python list â€” NOT an nn.Module â€” so nn.Module's
# __setattr__ treats it as a regular attribute in self.__dict__.
# ---------------------------------------------------------------------------
class _ModelProxy(list):
    """A plain list subclass. model.model[-1] returns the DummyHead."""
    pass


# ---------------------------------------------------------------------------
# UltralyticsAPGWrapper
#
# Conforms to the Ultralytics BaseModel contract verified against the
# installed Ultralytics 8.4.116 source code.
#
# Key interfaces implemented:
#
#   forward(x):
#     - x is dict (training)  â†’ self.loss(x) â†’ (loss_tensor, loss_dict)
#     - x is tensor (inference) â†’ (y_decoded, preds_dict) tuple
#
#   loss(batch, preds=None):
#     - Lazily inits criterion via init_criterion()
#     - Returns self.criterion(preds, batch) â†’ (loss_tensor, loss_dict)
#
#   init_criterion():
#     - Returns v8DetectionLoss(self)
#     - v8DetectionLoss reads: model.args, model.model[-1].{nc,reg_max,stride}
#
# Validator call path (the previous crash path):
#     preds = model(batch["img"])       â† tensor, eval mode â†’ (y, dict)
#     model.loss(batch, preds)          â† preds is tuple â†’ parse_output â†’ dict
#     postprocess(preds)                â† NMS on preds[0] = y tensor
# ---------------------------------------------------------------------------
class UltralyticsAPGWrapper(nn.Module):
    def __init__(self, nc, names):
        super().__init__()
        self.nc = nc
        self.names = names
        self.task = "detect"
        self.yaml = {"ch": 3, "nc": nc}
        self.reg_max = 16
        self.stride = torch.tensor([8.0, 16.0, 32.0])

        # The actual APG-ASR architecture (all parameters live here)
        self.apg = APG_ASR(num_classes=nc, reg_max=self.reg_max)

        # DFL for inference-time box decoding (frozen weights)
        self.dfl = DFL(self.reg_max)

        # model.model[-1] proxy for v8DetectionLoss.__init__
        # This is a plain Python list, NOT an nn.Module, so it goes
        # into self.__dict__ and doesn't interfere with named_parameters()
        dummy_head = DummyHead(nc, self.reg_max, self.stride)
        self.model = _ModelProxy([None, dummy_head])

        # Anchor cache for inference
        self._cached_anchors = None
        self._cached_strides = None
        self._cached_shape = None

    def __getattr__(self, name):
        # Support older checkpoints that didn't have self.yaml saved in their dict
        if name == "yaml":
            return {"ch": 3, "nc": getattr(self, "nc", 80)}
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def device(self):
        return next(self.parameters()).device

    def _format_predictions(self, preds_raw):
        """Convert raw APG-ASR outputs to Ultralytics dict format.

        preds_raw: list of 3 tensors [B, 144, H, W] from detection head
        Returns: {"boxes": [B, 64, N], "scores": [B, nc, N], "feats": [list]}
        """
        bs = preds_raw[0].shape[0]
        boxes_list = []
        scores_list = []

        for xi in preds_raw:
            # Each xi is [B, reg_max*4 + nc, H, W] = [B, 144, H, W]
            box_feat, cls_feat = xi.split((self.reg_max * 4, self.nc), 1)
            boxes_list.append(box_feat.reshape(bs, self.reg_max * 4, -1))
            scores_list.append(cls_feat.reshape(bs, self.nc, -1))

        return {
            "boxes": torch.cat(boxes_list, dim=-1).contiguous(),
            "scores": torch.cat(scores_list, dim=-1).contiguous(),
            "feats": [f.contiguous() for f in preds_raw],
        }

    def _decode_for_inference(self, preds_dict):
        """Decode predictions for NMS: DFL â†’ dist2bbox â†’ concat with sigmoid scores.

        Returns: [B, 4+nc, N] tensor for NMS postprocessing
        """
        feats = preds_dict["feats"]
        shape = feats[0].shape

        # Recompute anchors if spatial shape changed
        if self._cached_shape != shape:
            self._cached_anchors, self._cached_strides = (
                a.transpose(0, 1)
                for a in make_anchors(feats, self.stride, 0.5)
            )
            self._cached_shape = shape

        # DFL: [B, 64, N] â†’ [B, 4, N]
        dbox = self.dfl(preds_dict["boxes"])
        # dist2bbox: convert ltrb distances to xywh boxes, scale by stride
        dbox = dist2bbox(dbox, self._cached_anchors.unsqueeze(0), xywh=True, dim=1)
        dbox = dbox * self._cached_strides

        # Concat decoded boxes + sigmoid class scores â†’ [B, 4+nc, N]
        y = torch.cat((dbox, preds_dict["scores"].sigmoid()), dim=1)
        return y

    def fuse(self, *args, **kwargs):
        """Dummy fuse method for Ultralytics export compatibility."""
        return self

    def init_criterion(self):
        """Initialize the Ultralytics v8 detection loss.

        v8DetectionLoss.__init__ reads:
          - model.args (set by DetectionTrainer.set_model_attributes)
          - model.model[-1].{nc, reg_max, stride, no}
        """
        return v8DetectionLoss(self)

    def loss(self, batch, preds=None):
        """Compute loss. Matches BaseModel.loss() contract.

        Called by:
          - forward(dict) during training: preds=None â†’ runs forward(batch["img"])
          - validator: preds=(y_tensor, preds_dict) â†’ criterion.parse_output extracts dict
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])

        return self.criterion(preds, batch)

    def forward(self, x, *args, **kwargs):
        """Forward pass matching BaseModel contract.

        Training (x is dict):
          â†’ compute loss â†’ return (loss_tensor, loss_dict)

        Inference (x is tensor):
          â†’ return (y_decoded, preds_dict) tuple
            - y_decoded: [B, 4+nc, N] for NMS postprocessing
            - preds_dict: {"boxes", "scores", "feats"} for loss computation
        """
        if isinstance(x, dict):
            # Training path: compute forward + loss
            return self.loss(x)

        # Forward through APG-ASR
        preds_raw = self.apg(x)
        preds_dict = self._format_predictions(preds_raw)

        if self.training:
            # Compile-mode path: return dict for loss() to consume
            return preds_dict

        # Eval mode: decode for NMS + keep raw for loss
        y = self._decode_for_inference(preds_dict)
        return (y, preds_dict)


# ---------------------------------------------------------------------------
# APG_ASRTrainer: Minimal override of DetectionTrainer
# ---------------------------------------------------------------------------
class APG_ASRTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        import yaml
        with open(self.args.data, "r") as f:
            data_dict = yaml.safe_load(f)
        nc = data_dict.get("nc", 80)
        names = data_dict.get("names", {i: str(i) for i in range(nc)})
        
        model = UltralyticsAPGWrapper(nc, names)
        
        # EXPLICITLY LOAD WEIGHTS if provided, because Ultralytics often silently fails 
        # to match keys on custom wrappers.
        actual_weights = os.environ.get("APG_RESUME_WEIGHTS", None)
        if actual_weights and str(actual_weights).endswith('.pt'):
            ckpt = torch.load(actual_weights, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict):
                # The model in ckpt could be a state_dict or the full model object under 'model' or 'ema'
                if 'ema' in ckpt and ckpt['ema']:
                    state_dict = ckpt['ema'].state_dict() if hasattr(ckpt['ema'], 'state_dict') else ckpt['ema']
                elif 'model' in ckpt and ckpt['model']:
                    state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
                else:
                    state_dict = ckpt
                
                # Check if it was saved with 'model.' prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Our UltralyticsAPGWrapper state dict keys start with 'apg.' or 'dfl.'
                    clean_k = k.replace("model.", "") if k.startswith("model.") else k
                    if not clean_k.startswith("apg.") and not clean_k.startswith("dfl."):
                        clean_k = "apg." + clean_k
                    new_state_dict[clean_k] = v
                    
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                loaded = len(new_state_dict) - len(unexpected)
                total = len(model.state_dict())
                print(f"\n[INFO] ðŸ’¥ CRITICAL: Manually Transferred {loaded}/{total} items from {actual_weights}")
                if missing:
                    print(f"[INFO] Missing keys: {len(missing)}")
                    
        # Apply Phase-Specific Freezing Rules
        current_phase = os.environ.get("APG_PHASE", "1")
        if current_phase == "3":
            print("\n[INFO] Phase 3 detected (Stage 2: Controlled Re-entry). Freezing Transformer...")
            frozen_count = 0
            for name, param in model.named_parameters():
                # Freeze transformer branch
                if "transformer" in name:
                    param.requires_grad = False
                    frozen_count += 1
            print(f"[INFO] Transformer successfully frozen ({frozen_count} tensors). Only CNN + Gate + Neck + Head will train.")
            
        elif int(current_phase) >= 4:
            print(f"\n[INFO] Phase {current_phase} detected (Final Training Stages). UNFREEZING ALL components...")
            for name, param in model.named_parameters():
                param.requires_grad = True
            print("[INFO] Transformer is FULLY UNFROZEN. The entire network will train.")
            
        return model


# ---------------------------------------------------------------------------
# train_phase: Master function called by the phase runner
# ---------------------------------------------------------------------------
def train_phase(total_epochs: int, resume_path: str = None, is_true_resume: bool = False, phase: int = 1):
    """Launch a training run.

    Args:
        total_epochs: Total number of epochs for this run. Ultralytics handles
            start_epoch internally from the checkpoint when resuming.
        resume_path: Path to last.pt for resume. None for fresh start.
        is_true_resume: Whether to use 'resume' mode or just load 'model'.
        phase: The current phase number.
    """
    data_path = os.path.join(ROOT, "config", "dataset.yaml")

    overrides = {
        # Model (dummy â€” get_model overrides it)
        "model": "yolov8n.yaml",
        "data": data_path,
        # Training
        "epochs": total_epochs,
        "batch": 4,  # Reduced to 4 to prevent OOM on 6GB VRAM (Scenario 1)
        "imgsz": 640,
        "device": 0,
        "amp": True,
        "workers": 2, # Decreased to 2 to prevent RAM crash
        "cache": False,
        # Augmentation
        "mosaic": 1.0,
        "mixup": 0.0,
        "close_mosaic": 2,
        # Optimizer
        "optimizer": "AdamW",
        "lr0": 0.0001,  # Changed to 1e-4 for gentle fine-tuning of converged Phase 1 weights
        "weight_decay": 0.0005,
        "cos_lr": True,
        "warmup_epochs": 0, # Disable warmup to prevent momentum spikes on resume
        
        # Validation & saving
        "val": True,
        "save": True,
        "save_period": 5,
        # Output
        "project": ROOT,
        "name": "yolo_internal",
        "exist_ok": True,
        # Reproducibility
        "seed": 42,
    }

    if resume_path:
        os.environ["APG_RESUME_WEIGHTS"] = resume_path
        if is_true_resume:
            overrides["resume"] = resume_path
            # For true resume, we might still want to let Ultralytics handle optimizer state if possible
        else:
            # We don't set overrides["model"] = resume_path, because Ultralytics will crash trying to parse a naked state dict
            # We let our environment variable and get_model handle the weight injection.
            overrides["model"] = "yolov8n.yaml" # Dummy model to satisfy Ultralytics

    trainer = APG_ASRTrainer(overrides=overrides)
    trainer.train()

