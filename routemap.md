# CDAC Project Route Map 🗺️

Welcome to the APG-ASR (Adaptive Probabilistic Gated CNN–Transformer with Agentic Selective Re-Inference) project! 
This document serves as your GPS to navigate the codebase, understand the ML pipeline, and quickly find exactly what you're looking for.

---

## 🔄 The Pipeline (How Data Flows)

Our machine learning pipeline follows a clean, modular structure:

1. **Data Prep (`tools/`)**: We start by running standalone scripts (like `convert_coco_to_yolo.py` or `prepare_subset.py`) to process our raw datasets (e.g., COCO 2017) into trainable formats.
2. **Model Definition (`src/models/`)**: The architecture for our models, specifically the custom APG-ASR and YOLOv8 integrations, are defined here.
3. **Training (`src/training/`)**: Models are trained using scripts in this directory. 
4. **Experiment Tracking (`experiments/`)**: While training, all intermediate weights, logs, metrics, and checkpoints are aggressively saved here. This ensures we never lose progress and can reproduce any run.
5. **Finalizing (`models/`)**: Once an experiment yields a successful, final model, its weights (`.pt` file) are promoted to this dedicated directory. 
6. **Inference & Evaluation (`src/inference/` & `src/evaluation/`)**: We run our final models against test sets to gauge performance. The entry point for this is `run.py` at the project root.
7. **Outputs (`outputs/`)**: Final reports (like HTML summaries), telemetry logs, and evaluation metrics are dumped here for analysis.

---

## 📁 Folder Directory (Where is Where?)

Here is the exact breakdown of every top-level folder and what you will find inside.

### `models/` (The Deliverables)
**Purpose:** Stores the final, trained model weights ready for production or evaluation. No intermediate junk.
- `models/yolov8/base/`: Contains the original, pre-trained `yolov8n.pt`.
- `models/yolov8/custom/`: Contains our baseline trained YOLOv8 model (`yolov8_custom_best.pt`).
- `models/apg_asr/weights/`: Contains the star of the show — our trained APG-ASR models (`apg_asr_best.pt`, `apg_asr_phase1.pt`).

### `src/` (The Brains)
**Purpose:** The core python package containing all source code for the pipeline.
- `src/models/`: Neural network architecture definitions.
- `src/training/`: Scripts that handle the training loops, optimizers, and loss functions.
- `src/inference/`: Scripts to run predictions on new data (houses `run_inference.py`).
- `src/evaluation/`: Scripts to calculate metrics (mAP, F1-Score, etc.) against test sets.
- `src/utils/`: Reusable helper functions (like metric calculators or data loaders) used across `src/`.

### `experiments/` (The Laboratory)
**Purpose:** The messy, highly-detailed scratchpad where training runs live.
- `experiments/yolo_runs/`: Automatically generated directories from YOLOv8 training/validation commands (contains charts, confusion matrices, and intermediate `weights/`).
- `experiments/apg_asr/` & `experiments/baseline_yolov8/`: Detailed checkpoints (`best.pt`, `last.pt`) and run artifacts for our specific models.

### `outputs/` (The Results)
**Purpose:** Storing the artifacts generated *after* training and evaluation.
- `outputs/predictions/`: Visual predictions and output images from running inference on new data.
- `outputs/logs/`: Telemetry and console logs (e.g., `telemetry.log`).
- `outputs/results/`: JSON/CSV metric dumps (e.g., `harness_c_results.json`).
- `outputs/reports/`: Formatted visual reports (e.g., `paper_temp.html`).

### `research/` (The Literature)
**Purpose:** Everything related to the academic paper and background research.
- `research/submission/`: The final, polished files ready for journal/mentor submission.
- `research/drafts/`: Older iterations and scratchpads for the paper.
- `research/references/`: Reference materials and prior work.
- `research/reviews/`: Feedback and review documents.

### `tools/` (The Workbench)
**Purpose:** Standalone, one-off scripts that don't belong in the main pipeline. 
- Contains dataset converters (`convert_coco_to_yolo.py`) and report generators (`write_html.py`).

### `assets/` & `docs/`
- **`assets/`**: Static files, images, or pre-trained base weights used generically.
- **`docs/`**: Markdown documentation, meeting notes, and architecture diagrams.

### Root Files
- **`run.py`**: The main entry point to execute the inference pipeline. Run this to test the system!
- **`README.md`**: The high-level executive summary of the project.
- **`requirements.txt`**: The python dependencies required to run the code.
