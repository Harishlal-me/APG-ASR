To realistically achieve

Metric	Target
mAP@50	0.57–0.59
mAP@50:95	0.41–0.44
Precision	0.66–0.69
Recall	0.51–0.55

you need much more than just replacing the backbone. The improvements come from the combination of architecture, training, and evaluation.

Stage 1 — Strong Baseline ✅ (Completed)

You have already done this.

YOLOv8n
100 epochs
AdamW
COCO 50K

mAP50      : 0.494
mAP50-95   : 0.343

This is your reference.

Stage 2 — Build APG-ASR Architecture

Replace only the backbone.

Input
      │
      ▼
CNN Stem
      │
 ┌────┴────┐
 │         │
 ▼         ▼
CNN     Transformer
 │         │
 └────┬────┘
      │
Adaptive Probabilistic Gate
      │
Weighted Feature Fusion
      │
PAN-FPN
      │
YOLO Head
CNN
Use YOLOv8 CSPDarknet

Transformer
RTX 5060 (High GPU Laptop)
Swin-T (Swin Transformer Tiny)

### WHY EACH COMPONENT EXISTS

CNN (CSPDarknet)
→ extracts local spatial features efficiently

Transformer (Swin-T)
→ captures global dependencies and context

Adaptive Gate
→ dynamically balances CNN vs Transformer contributions

Feature Fusion
→ learns optimal combination instead of static concatenation

PAN-FPN
→ multi-scale feature aggregation

YOLO Head
→ efficient detection output

### FLOW WITH SHAPES

Input: 1×3×640×640

CNN Output:
P3: 1×256×80×80
P4: 1×512×40×40
P5: 1×1024×20×20

Transformer Output:
Resized to match CNN scales

Gate Output:
Weighted combination (same dimensions)

Final:
Fed into PAN-FPN
Stage 3 — Adaptive Probabilistic Gate

This is your novelty.

Instead of

CNN
+
Transformer

You use

Gate

↓

CNN gets
60%

Transformer gets
40%

or

20%

80%

depending on the image.

Implement using

Gumbel Softmax
Stage 4 — Feature Fusion

Don't simply concatenate.

Instead

CNN Feature

↓

Weight α

+

Transformer Feature

↓

Weight β

↓

Fusion

Learn α and β during training.

Stage 5 — Neck

Keep

YOLO PAN-FPN

Don't reinvent this.

Stage 6 — Detection Head

Keep

YOLOv8 Head
Stage 7 — Better Training

### EXACT TRAINING STRATEGY

Training Strategy

Day 1: Epoch 1–25
- Validate stability
- Monitor loss trend

Day 2: Epoch 26–50
- Observe mAP growth
- Check overfitting

Day 3: Epoch 51–75
- Fine convergence
- Stabilization phase

Day 4: Epoch 76–100
- Final optimization
- Best model selection

### MONITORING RULES

If training loss not decreasing → LR too high

If validation loss increases → overfitting

If mAP stagnant → architecture issue

If GPU usage < 80% → dataloader bottleneck
Stage 8 — Hyperparameter Tuning

Your baseline probably wasn't fully tuned.

Experiment with:

Learning Rate

Weight Decay

Warmup

Label Smoothing

Batch Size

Image Size

Augmentation

This alone can sometimes improve mAP by 1–3 percentage points.

Stage 9 — Agentic AI

After training finishes

NOT another training.

Instead

Detection

↓

Confidence Checker

↓

If confidence > threshold

Return prediction

Else

↓

Agent

↓

Analyze image

↓

Blur?

Brightness?

Occlusion?

Small object?

↓

Choose tool

↓

Re-inference

↓

Return improved result

This is coding and orchestration rather than training a separate detector.

Stage 10 — Evaluate Properly

### QUALITATIVE ANALYSIS

Where APG-ASR improves:
- Small objects
- Blurry images
- Crowded scenes
- Low light

Where it fails:
- extreme occlusion
- very tiny objects

### ADD VISUAL COMPARISON

Baseline Miss → APG-ASR Detect

Run

Baseline

↓

APG-ASR

↓

Gate Removed

↓

Transformer Removed

↓

Agent Removed

These are your ablation studies.

Stage 11 — Compare Everything

Don't compare only mAP.

Include:

mAP50

mAP50-95

Precision

Recall

F1 Score

FPS

GFLOPs

Parameters

Training Time

Inference Time

Memory Usage

### ABLATION INTERPRETATION

Without Transformer → drop in global context → worse small object detection

Without Gate → fixed weighting → less adaptability

Without Agent → no improvement in difficult cases
Stage 12 — Error Analysis

Show

Baseline misses

↓

APG-ASR detects

For example:

Small objects
Blurry images
Crowded scenes
Low-light images

These qualitative examples strengthen your paper.

Expected Contributions to Performance

### CONTRIBUTION BREAKDOWN

Expected Gains Breakdown

Swin-T (RTX 5060) → +3–5% mAP50-95
Adaptive Gate → +1–2%
Feature Fusion → +0.5–1%
Hyperparameter tuning → +1–3%
Agentic AI → +0.5–1.5%

Total Expected (non-linear): +6–9%

### VALIDATION LOGIC

Validation Plan

If Transformer removed → drop in small object detection

If Gate removed → reduced adaptability across scenes

If Fusion replaced with concat → slight performance drop

If Agent removed → worse performance on low-confidence cases

What success looks like
Metric	Baseline	Target APG-ASR
mAP@50	0.494	0.57–0.59
mAP@50:95	0.343	0.41–0.44
Precision	0.60	0.66–0.69
Recall	0.46	0.51–0.55
Parameters	3.15M	12–15M
GFLOPs	8.9	20–24
FPS	384	180–240

If you achieve results in that range and support them with proper ablation studies, agent evaluation, and reproducible experiments, you'll have a technically solid APG-ASR project suitable for your minor project and a reasonable basis for submitting to student-focused conferences or workshops.