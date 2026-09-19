# APG-ASR: Adaptive Probabilistic Gated CNN–Transformer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange)
![Status](https://img.shields.io/badge/Status-Active_Research-success)

**Adaptive Probabilistic Gated CNN–Transformer with Agentic Selective Re-Inference** is a novel architecture designed to enhance the accuracy and efficiency of object detection systems. This repository contains the official PyTorch implementation, experimental setups, and research documents related to this proposed method.

## 📌 Overview

Traditional object detection frameworks often struggle with optimizing the trade-off between computational efficiency and detection accuracy, particularly in complex or noisy environments. APG-ASR introduces:
1. **Adaptive Probabilistic Gated CNN-Transformer**: A hybrid backbone that intelligently routes features based on spatial complexity.
2. **Agentic Selective Re-Inference**: An agent-driven mechanism that selectively re-processes high-uncertainty regions to refine predictions dynamically.

## 📂 Repository Structure

The repository is strictly organized to isolate datasets, utilities, and individual experiments:

- **`assets/`** - Static assets for documentation (images, architecture diagrams, logos).
- **`datasets/`** - Training, validation, and testing datasets.
  - `coco_original/` - Full COCO 2017 dataset.
  - `coco_50k/` - Subsampled COCO 50K subset used for rapid benchmarking.
- **`docs/`** - Project documentation, meeting notes, and reference materials.
- **`experiments/`** - Isolated environments for training runs, model checkpoints, and metric evaluations.
- **`paper/`** - LaTeX source files, figures, and drafts for the final research publication.
- **`tools/`** - Reusable utility scripts for dataset preparation, subsetting, and format conversions.

## 🧪 Current Experiments

| Experiment | Status | Description |
| :--- | :--- | :--- |
| **`baseline_yolov8`** | ✅ Frozen | Baseline YOLOv8n trained for 100 epochs on COCO 50K. |
| **`apg_asr`** | 🚧 Development | Proposed Adaptive Probabilistic Gated CNN–Transformer. |
| **`apg_asr_ablation_gate`** | ⏳ Planned | Ablation study: Removing probabilistic gate. |
| **`apg_asr_ablation_agent`** | ⏳ Planned | Ablation study: Removing agentic selective re-inference. |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.3.1+ (with CUDA 12.1 support)
- Ultralytics 8.4.115

Install the required dependencies from the root directory:
```bash
pip install -r requirements.txt
```

*(Further instructions on training, validation, and inference will be added as the APG-ASR architecture reaches its first stable release.)*

## 🧑‍🔬 Author

**Harishlal**  
*SRM Institute of Science and Technology (SRM IST)*

---
*This repository is currently under active development as part of an ongoing research project.*
