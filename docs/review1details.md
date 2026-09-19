# Review 1 Needs (Filled for APG-ASR)

## Slide 1 – Title Slide
1. **Project Title**: APG-ASR: Adaptive Probabilistic Gated CNN–Transformer with Agentic Selective Re-Inference
2. **Team Members**: 
   - Harishlal (Register Number:RA2411003010228 )
3. **Guide Name**: Dr. Ragunthar T*
4. **Department**: C.Tech*
5. **Review Date**: 07.08.26*

## Slide 2 – Project Overview
1. **One-line project description**: APG-ASR is a novel object detection framework that uses an intelligent gating mechanism to route features between CNNs and Transformers, coupled with an AI agent that selectively re-evaluates uncertain detections to boost accuracy without sacrificing speed.
2. **Domain**: Computer Vision, Deep Learning, Agentic AI
3. **Target Users**: Autonomous vehicles, Smart surveillance systems, and AI Researchers requiring high accuracy in complex environments.
4. **Final Deliverable**: Research prototype and evaluation framework.

## Slide 3 – Problem Statement & Motivation
1. **What exact problem are you solving?**: Balancing computational efficiency with detection accuracy in complex, noisy, or crowded visual environments.
2. **Why do existing detectors fail?**: Pure CNNs struggle with global context, while pure Transformers (ViTs) are computationally too heavy for real-time edge use. Static hybrid models waste compute by processing simple background regions through expensive Transformer blocks.
3. **Why is solving this important?**: Real-time systems like autonomous driving need high accuracy on difficult objects (e.g., heavily occluded pedestrians) but must process the rest of the frame extremely fast.
4. **Any real-world applications?**: Autonomous driving perception, drone surveillance, and automated defect inspection.

## Slide 4 – Research Gap & Existing Solutions
- **Existing Solution 1**
  - **Name**: YOLOv8 (Ultralytics)
  - **Strength**: Extremely fast inference and highly optimized CNN architecture.
  - **Weakness**: Lacks global context modeling; struggles with complex spatial relationships and heavy occlusions.
- **Existing Solution 2**
  - **Name**: Vision Transformers (ViT) / DETR
  - **Strength**: Excellent at capturing long-range dependencies and global context.
  - **Weakness**: Computationally expensive, high memory footprint, and slow inference.
- **Existing Solution 3**
  - **Name**: Standard Hybrid Architectures (e.g., CNN + Transformer blocks)
  - **Strength**: Balances local feature extraction with global context.
  - **Weakness**: Applies both processing paths statically to all regions, leading to wasted computational resources on simple image areas.
- **Your Novel Contribution**: 
  - **Adaptive Probabilistic Gate**: Dynamically decides (pixel-by-pixel or patch-by-patch) whether to use a cheap CNN or an expensive Transformer based on the region's complexity.
  - **Agentic Selective Re-Inference**: An AI agent that looks at the initial predictions and says, "I'm not confident about this specific region," crops it, enhances it, and re-runs inference just on that small crop to refine the result.

## Slide 5 – Objectives
- **Objective 1**: Develop a hybrid CNN-Transformer backbone equipped with an adaptive probabilistic gate to dynamically route spatial features.
- **Objective 2**: Implement an agentic selective re-inference module to evaluate prediction uncertainty and refine bounding boxes.
- **Objective 3**: Benchmark the architecture against state-of-the-art models (like YOLOv8) to demonstrate a superior accuracy-to-compute trade-off.

**Success Criteria**
- Better mAP@50-95 compared to the YOLOv8n baseline on the COCO subset.
- Lower average FLOPs compared to a full Transformer-based detector.
- Improved detection confidence on heavily occluded or small objects.

## Slide 6 – Scope
**In Scope**: 
- CNN and Transformer backbone design.
- YOLO detection head integration.
- Adaptive Gating mechanism.
- Uncertainty-based Agentic Re-inference.
- Evaluation on the COCO 2017 (50K subset) dataset.

**Out of Scope**: 
- Video object detection (temporal tracking).
- Mobile or Edge device deployment (e.g., TensorRT/TFLite export).
- Multi-camera 3D detection.

## Slide 7 – Architecture / Methodology
**Pipeline Flow**:
Input Image ↓ 
CNN Stem (Initial low-level features) ↓ 
**Adaptive Probabilistic Gate** (Decides routing per region) ↓ 
*Split Path*: [CNN Branch (Simple regions)] OR [Transformer Branch (Complex regions)] ↓ 
Feature Pyramid Network (FPN) for multi-scale fusion ↓ 
YOLO Detection Head ↓ 
Initial Bounding Boxes & Confidence Scores ↓ 
**Agentic Uncertainty Check** (Is the confidence too low / variance too high?) ↓ 
*If Yes*: Crop Region → Local Image Enhancement → Re-Inference → Refined Prediction ↓ 
Final Output.

**Why did you choose this architecture?**: 
It directly solves the compute-vs-accuracy dilemma. The gate saves compute on easy regions (like sky or road), while the Agentic Re-inference ensures we don't miss hard objects (like a partially hidden pedestrian), mimicking human visual attention.

## Slide 8 – Dataset, Tools & Evaluation
**Dataset**: 
- **Name**: MS COCO 2017 (Using a customized 50K subset for rapid iteration)
- **Size**: 50,000 images total.
- **Classes**: 80 object classes.
- **Splits**: Train: 40,000 | Validation: 5,000 | Test: 5,000
- **Preparation**: Pre-processed and converted to YOLO format using custom Python tools.

**Tools**: 
- **Languages/Frameworks**: Python, PyTorch (2.3.1+), Ultralytics (8.4.115).
- **Hardware**: NVIDIA RTX 4050 Laptop GPU (6GB).
- **Version Control**: Git.

**Evaluation Metrics**: 
- Mean Average Precision (mAP@50 and mAP@50-95).
- Inference Time / Latency (ms).
- Computational Cost (FLOPs).

## Slide 9 – Risks & Ethics
- **Biggest technical risks**: The probabilistic gate might not converge easily during training, or the re-inference agent might slow down the overall pipeline too much.
- **Fallback plans**: If the gate fails to converge, we will use a static spatial attention map. If re-inference is too slow, we will limit it to only the top-1 most uncertain region per image.
- **Ethical concerns & Limitations**: The model's performance relies heavily on the COCO dataset's biases. It may underperform on underrepresented demographics or novel environments not present in the training data.

## Slide 10 – Work Completed
- **Dataset**: COCO 50K subset successfully prepared, formatted, and validated.
- **Repository Setup**: Fully structured research environment established (`datasets/`, `experiments/`, `tools/`).
- **Baseline Training**: Standard YOLOv8n baseline model fully trained for 100 epochs on the 50K dataset.
- **Evaluation Framework**: Scripts written and verified for loading models, running inference, and calculating metrics.
- **Documentation**: Initial README and architectural concepts outlined.

## Slide 11 – Timeline
- **Week 1**: Dataset preparation, environment setup, and Baseline YOLOv8 training (Completed).
- **Week 2**: Implementation of the CNN-Transformer hybrid backbone and the Adaptive Probabilistic Gate.
- **Week 3**: Development of the Agentic Selective Re-inference module.
- **Week 4**: Final integration, hyperparameter tuning, evaluation against the baseline, and report writing.

## Slide 12 – Plan Before Review 2
**Next Deliverables**:
- Complete the PyTorch implementation of the Adaptive Probabilistic Gate.
- Integrate the hybrid backbone into the YOLO framework and ensure it successfully trains without exploding gradients.
**Approvals Needed**:
- Validation of the proposed architectural flowchart by the guide.

## Slide 13 – Conclusion
- **Expected Contributions**: A novel, scalable object detection framework that achieves higher accuracy on occluded and complex scenes while consuming less average computational power than pure Vision Transformer models.
- **Future Work (if any)**: Extending the Agentic Selective Re-inference module to handle temporal tracking in video streams, and exploring lightweight deployment for edge devices.
- **Key Takeaways**: Intelligent resource allocation (adaptive gating) and targeted re-evaluation (agentic re-inference) offer a highly effective path to breaking the accuracy-vs-efficiency trade-off in modern computer vision systems.

---

### Exact Mapping (Reference Checklist)

| PPT Slide | What I'll Ask You |
| :--- | :--- |
| Slide 1 | Project title, team members, guide, date |
| Slide 2 – Project Overview | One-line description, target user, domain, team roles, one-month deliverable |
| Slide 3 – Problem Statement & Motivation | Problem, motivation, evidence (papers/interviews/dataset), impact |
| Slide 4 – Research Gap & Existing Solutions | Existing methods, strengths, limitations, your improvement |
| Slide 5 – Objectives & Success Criteria | Three objectives and measurable success criteria |
| Slide 6 – Project Scope | In scope, out of scope, minimum viable prototype |
| Slide 7 – Proposed Architecture / Methodology | Complete pipeline and design decisions |
| Slide 8 – Dataset, Tools & Evaluation | Dataset details, tools, frameworks, metrics, readiness evidence |
| Slide 9 – Risks, Ethics & Mitigation | Risks, fallback plans, ethics |
| Slide 10 – Work Completed | Everything you've actually completed so far |
| Slide 11 – Timeline | Review roadmap and milestones |
| Slide 12 – Plan Before Review 2 | Next deliverables, approvals needed, owners and deadlines |
| Slide 13 | The template uses this as "Plan Before Review 2" (there isn't a separate conclusion slide in your template). |
| Slide 14 | Presentation checklist. |
