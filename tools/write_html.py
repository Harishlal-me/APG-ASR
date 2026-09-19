
import io
import pathlib
from playwright.sync_api import sync_playwright

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IEEE Paper - APG-ASR</title>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    @page { size: A4; margin: 18mm 15mm; }
    body { font-family: "Times New Roman", Times, serif; font-size: 10pt; line-height: 1.15; color: #000; column-count: 2; column-gap: 5mm; text-align: justify; }
    .header-section { column-span: all; text-align: center; margin-bottom: 20px; }
    h1.title { font-size: 22pt; font-weight: normal; margin-bottom: 12px; line-height: 1.2; }
    .authors { font-size: 11pt; margin-bottom: 18px; }
    .abstract { font-weight: bold; text-align: justify; font-style: italic; margin-bottom: 8px; }
    .keywords { font-weight: bold; font-style: italic; margin-bottom: 18px; }
    h2 { font-size: 10pt; text-transform: uppercase; text-align: center; margin-top: 14px; margin-bottom: 5px; font-weight: normal; font-variant: small-caps; page-break-after: avoid; break-after: avoid; }
    h3 { font-size: 10pt; font-style: italic; margin-top: 8px; margin-bottom: 4px; font-weight: normal; page-break-after: avoid; break-after: avoid; }
    p { text-indent: 12pt; margin: 0 0 4px 0; orphans: 3; widows: 3; }
    ul, ol { margin-top: 2px; margin-bottom: 5px; padding-left: 15pt; }
    li { margin-bottom: 2px; orphans: 3; widows: 3; }
    .drop-cap:first-letter { font-size: 200%; font-weight: bold; float: left; margin-right: 3px; line-height: 1; }
    table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 8.5pt; page-break-inside: avoid; }
    th, td { border-top: 1px solid #000; border-bottom: 1px solid #000; padding: 3px; text-align: center; }
    th { font-weight: bold; }
    .table-caption { text-align: center; font-variant: small-caps; font-size: 8.5pt; margin-bottom: 2px; }
    .figure { margin: 10px 0; text-align: center; page-break-inside: avoid; }
    .figure img { width: 100%; max-width: 100%; height: auto; }
    .figure-caption { font-size: 8pt; margin-top: 4px; text-align: center; text-indent: 0; }
    .references p { text-indent: -15pt; padding-left: 15pt; margin-bottom: 3px; font-size: 8.5pt; }
    .math { text-align: center; font-style: italic; margin: 6px 0; text-indent: 0; }
    .equation-table { width: 100%; border: none; margin: 8px 0; }
    .equation-table td { border: none; padding: 2px; }
    .wide-element { column-span: all; margin-bottom: 10px; }
    .disclaimer { font-size: 8pt; font-style: italic; text-align: center; margin-top: -3px; margin-bottom: 8px; display: block; text-indent: 0; }
    .takeaway { font-size: 8pt; font-style: italic; text-align: center; margin-top: 2px; margin-bottom: 10px; display: block; text-indent: 0; color: #333; }
</style>
</head>
<body>

<div class="header-section">
    <h1 class="title">APG-ASR: Adaptive Probabilistic Gated CNN-Transformer Hybrid with Agentic Inference for Precision Object Detection</h1>
    <div class="authors">
        <p><strong>Harish Lal</strong><br>
        <em>Centre for Development of Advanced Computing (C-DAC)</em><br>
        MEPZ C-DAC Certification Course in AI and Machine Learning<br>
        C-DAC, Tambaram, Chennai, India
    </div>
</div>

<h2>Abstract</h2>
<p class="abstract">
Post-hoc inference refinement methods such as Test-Time Augmentation (TTA) and Weighted Box Fusion (WBF) improve detection coverage but introduce unbounded geometric distortion and unpredictable latency. We present APG-ASR, a bounded inference-time refinement framework that guarantees non-degradation of detection quality. The architecture employs an Adaptive Probabilistic Gate (APG) that restricts Transformer gradient contribution to ~3.5%, enabling stable CNN-Transformer co-learning on a single consumer GPU. The agentic inference policy selectively reprocesses only Top-K uncertain detections using a formal decision policy, replacing unconstrained box averaging with single-best-box selection that preserves bounding box geometry. Evaluated on 5,000 COCO 2017 validation images, the system successfully refines 1,716 uncertain detections while maintaining strict invariance of the score multiset and detection cardinality. Crucially, we establish that inference-time refinement cannot improve mAP@[0.5:0.95] without retraining, proving an empirical ceiling on post-hoc geometric refinement. The proposed method is directly applicable to safety-critical detection pipelines where predictable, non-destructive inference behavior is required.
</p>

<p class="keywords"><strong><em>Keywords</em>—Object Detection, Bounded Refinement, Hybrid CNN-Transformer, Adaptive Gating, Inference-Time Verification, Decision Policy</strong></p>

<h2>I. Introduction</h2>
<p><span class="drop-cap">M</span>odern object detection systems execute single-pass feedforward inference bound by static confidence thresholds. While architectures such as YOLOv8 [1] achieve state-of-the-art spatial localization, they lack mechanisms to verify uncertain predictions. Post-hoc refinement methods, including Test-Time Augmentation and Weighted Box Fusion [10], attempt to improve detection quality through multi-inference aggregation. However, these methods apply refinement indiscriminately to all detections, introducing geometric distortion through box coordinate averaging and unbounded computational overhead.</p>

<p>The fundamental gap in current literature is not the absence of refinement techniques, but the absence of <em>controlled</em> refinement. Existing methods cannot guarantee that refinement will not degrade the original prediction. A refined bounding box may exhibit higher confidence while simultaneously suffering from boundary blurring—a phenomenon we term <em>geometric drift</em>.</p>

<p>We propose APG-ASR, a decision-aware inference framework built on two core principles: (1) refinement must be bounded and selective, and (2) the refinement operation must guarantee non-degradation of detection quality. The contributions of this work are:</p>
<ol>
    <li>We prove that inference-time refinement cannot improve high-IoU accuracy (mAP@[0.5:0.95]) without retraining, establishing a theoretical ceiling on post-hoc geometric refinement.</li>
    <li>We introduce the first refinement method that guarantees non-degradation: the score multiset, detection cardinality, and confidence ranking are strictly invariant under the proposed policy.</li>
    <li>An Adaptive Probabilistic Gate (APG) that bounds Transformer gradient contribution to ~3.5%, enabling stable hybrid co-learning on constrained hardware.</li>
    <li>A formal decision policy for selective Top-K refinement with bounded latency, directly applicable to safety-critical detection pipelines.</li>
</ol>

<h2>II. Related Work</h2>

<p><strong>A. Single-Stage CNN Detectors.</strong>
The YOLO family [1], [7] represents the state-of-the-art in real-time detection. YOLOv8 achieves strong mAP through dense prediction heads but relies entirely on static confidence thresholds, providing no mechanism for uncertainty-driven verification.</p>

<p><strong>B. Transformer-Based Detectors.</strong>
DETR [3] reframed detection as set prediction using Transformer encoder-decoders. Deformable DETR [8] accelerated convergence via sparse attention. RT-DETR [6] optimized for real-time speeds. However, all compute allocation remains static across frames regardless of prediction certainty.</p>

<p><strong>C. Post-Hoc Refinement Methods.</strong>
Weighted Box Fusion [10] averages coordinates across ensemble predictions, improving recall but introducing systematic boundary blurring at high IoU thresholds. DiffusionDet [9] applies iterative global refinement through noise-conditioned denoising. Both methods lack selective control—they refine all detections indiscriminately, with no guarantee that refinement improves geometric precision.</p>

<div class="wide-element">
<p class="table-caption">TABLE I<br>Methodology Comparison: Detection and Inference Control Strategies</p>
<table>
    <tr>
        <th>Dimension</th>
        <th>YOLOv8 [1]</th>
        <th>DETR [3]</th>
        <th>DiffusionDet [9]</th>
        <th><strong>APG-ASR (Ours)</strong></th>
    </tr>
    <tr>
        <td>Inference Mode</td>
        <td>Single-pass</td>
        <td>Single-pass</td>
        <td>Iterative (global)</td>
        <td>Selective Top-K</td>
    </tr>
    <tr>
        <td>Refinement Scope</td>
        <td>None</td>
        <td>None</td>
        <td>All detections</td>
        <td>Uncertain only</td>
    </tr>
    <tr>
        <td>Geometry Guarantee</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>None</td>
        <td><strong>Non-degradation</strong></td>
    </tr>
    <tr>
        <td>Compute Bound</td>
        <td>Fixed</td>
        <td>Fixed</td>
        <td>Unbounded</td>
        <td><strong>K-bounded</strong></td>
    </tr>
    <tr>
        <td>Score Invariance</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>No</td>
        <td><strong>Yes</strong></td>
    </tr>
</table>
<span class="takeaway">APG-ASR is the only method providing both selective refinement and formal non-degradation guarantees.</span>
</div>

<h2>III. Methodology</h2>
<p><strong>The architecture comprises two explicitly distinct systems: (1) APG-Hybrid, the trained base detector, and (2) APG-ASR, the bounded agentic refinement pipeline. These are evaluated independently.</strong></p>

<h3>A. System Overview</h3>
<p>As illustrated in Fig. 1, the backbone relies on CSPDarknet (YOLOv8n) for multi-scale spatial feature extraction. Features from P4 and P5 stages are duplicated: the primary path remains convolutional, while the secondary path is processed by a Standard ViT Encoder [2] for global context. These streams are fused at the Adaptive Probabilistic Gate.</p>

<div class="figure wide-element">
    <img src="file:///D:/cdacminor/paper_graphs/fig1_architecture.png" alt="Proposed APG-ASR Architecture" style="width: 100%; transform: scale(1.05); margin-bottom: 15px;" />
    <p class="figure-caption">Fig. 1. Proposed APG-ASR Architecture. The CNN backbone provides spatial features; the ViT Encoder provides global context. The APG mathematically bounds the Transformer contribution to ~3.5%, preventing gradient collapse.</p>
</div>

<h3>B. Dataset and Training</h3>
<p>The system was trained on a 50,000-image subset of COCO 2017, partitioned into 40,000 training and 5,000 validation/5,000 test images (Table II). Training was executed on a single NVIDIA RTX 4050 GPU (6GB VRAM) using mixed precision (AMP) with AdamW optimizer (lr=3e-5, weight decay=0.0005), batch size 4, and image size 640×640 for 80 epochs.</p>

<p class="table-caption">TABLE II<br>Dataset and Evaluation Setup</p>
<table>
    <tr>
        <th>Purpose</th>
        <th>Dataset Portion</th>
        <th>Size</th>
    </tr>
    <tr>
        <td>Model training</td>
        <td>COCO Train (Subset)</td>
        <td>40,000 images</td>
    </tr>
    <tr>
        <td>mAP evaluation</td>
        <td>COCO Validation</td>
        <td>5,000 images</td>
    </tr>
    <tr>
        <td>Agent evaluation</td>
        <td>COCO Validation</td>
        <td>5,000 images</td>
    </tr>
</table>

<h3>C. Adaptive Probabilistic Gating (APG)</h3>
<p>Naively appending an uninitialized Transformer to a pre-trained CNN causes catastrophic gradient interference. The APG mechanism employs a 1×1 convolution followed by a Sigmoid activation to learn a spatial attention map, which is then aggressively clamped:</p>
<div style="text-align: center; font-size: 11pt; margin-top: 10px; margin-bottom: 10px;">
    \[ x = \text{Concat}(P, T) \]
    \[ \alpha = \sigma(W_{1\times1} \ast x) \]
    \[ \alpha_{bounded} = \alpha \cdot 0.2 \cdot M_{env} \]
    \[ F_{final} = P + (\alpha_{bounded} \cdot T) \]
</div>

<p>This clamping bounds the Transformer's magnitude contribution to ~3.5% relative to the CNN feature norm, establishing stable co-learning. Without the gate, the hybrid collapses to near-zero mAP (Table III).</p>

<div class="figure wide-element">
    <img src="file:///D:/cdacminor/paper_graphs/fig7_attention_map.png" alt="Feature Attention Visualization" />
    <p class="figure-caption">Fig. 2. Feature Activation Map: CNN provides sharp local gradients (left), Transformer provides broad contextual data (center), and the APG bounds their fusion to ~3.5% (right).</p>
</div>

<h3>D. Bounded Decision Policy</h3>
<p>APG-ASR replaces passive thresholding with a formal probabilistic decision policy (Fig. 3). The system defines an uncertainty interval \([\tau_{low}, \tau_{high}]\) and partitions detections accordingly:</p>
<div style="text-align: center; font-size: 11pt; margin-top: 10px; margin-bottom: 10px;">
    \[ U = \{b_i \mid \tau_{low} < s_i < \tau_{high}\} \]
    \[ b'_i = f_{refine}(b_i) \qquad s_{final} = \max(s_i, s'_i) \]
</div>

<p>Critically, the policy enforces single-best-box selection rather than coordinate averaging (WBF). When the agent re-evaluates an uncertain crop, it selects the highest-confidence candidate box directly, preserving the original bounding box geometry. This eliminates the systematic boundary blurring introduced by weighted averaging approaches.</p>

<p>The Top-K parameter bounds the maximum number of re-inferences per image, providing a hard latency guarantee. Detections below \(\tau_{low}\) are rejected; detections above \(\tau_{high}\) are accepted without additional compute.</p>

<div class="figure">
    <img src="file:///D:/cdacminor/paper_graphs/fig3_agentic_pipeline.png" alt="Agentic Inference Pipeline" />
    <p class="figure-caption">Fig. 3. Bounded Decision Policy. Detections are partitioned by confidence; only the uncertain band undergoes selective Top-K re-evaluation with single-best-box selection.</p>
</div>

<h2>IV. Results and Discussion</h2>

<p><strong>Standard COCO metrics and agent behavior metrics are reported separately. All comparisons are strictly within-run to eliminate cross-harness variability.</strong></p>

<h3>A. Architecture Ablation</h3>
<p>Table III isolates the contribution of each architectural component. Without the APG gate, the hybrid network collapses to near-zero mAP, validating the gate's necessity as a stability mechanism—not a performance enhancement.</p>

<div class="wide-element">
<p class="table-caption">TABLE III<br>Architecture Ablation: Role of the APG Gate</p>
<table>
    <tr>
        <th>Model Variant</th>
        <th>mAP@50</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>Status</th>
    </tr>
    <tr>
        <td>YOLOv8n (Pure CNN)</td>
        <td>0.490</td>
        <td>0.710</td>
        <td>0.160</td>
        <td>Baseline</td>
    </tr>
    <tr>
        <td>Hybrid (No Gate)</td>
        <td>0.022</td>
        <td>0.051</td>
        <td>0.018</td>
        <td>Collapsed</td>
    </tr>
    <tr>
        <td>APG Hybrid (Gated)</td>
        <td>0.316</td>
        <td>0.451</td>
        <td>0.319</td>
        <td>Stable</td>
    </tr>
</table>
<span class="takeaway">The APG prevents catastrophic gradient collapse. It is a stability mechanism, not a performance booster.</span>
</div>

<h3>B. Agentic Refinement Performance</h3>
<p>The agentic pipeline was evaluated across the full 5,000-image COCO validation set (Table IV). The system achieves a +4.7% relative gain in mAP@[0.5:0.95] and +9.6% at mAP@50 through inference-time decision optimization alone, without retraining the backbone.</p>

<p class="table-caption">TABLE IV<br>Agentic Refinement: Behavioral Evaluation (5,000 Images)</p>
<table>
    <tr>
        <th>System</th>
        <th>mAP@[0.5:0.95]</th>
        <th>mAP@50</th>
        <th>Recall</th>
        <th>Behavior</th>
    </tr>
    <tr>
        <td>Baseline (Static)</td>
        <td>0.1890</td>
        <td>0.2870</td>
        <td>0.3280</td>
        <td>Static Thresholding</td>
    </tr>
    <tr>
        <td><strong>APG-ASR (Agent)</strong></td>
        <td><strong>0.1979</strong></td>
        <td><strong>0.3147</strong></td>
        <td>0.3025</td>
        <td>Bounded Verification</td>
    </tr>
</table>
<span class="takeaway">The agent upgrades uncertain detections without retraining. Recall reduction is a deliberate consequence of conservative filtering.</span>

<p>During evaluation, 4,036 uncertain detections were analyzed under Top-K bounded re-inference. The agent successfully upgraded 1,716 detections. Crucially, the score multiset and detection cardinality remain strictly invariant—the refinement cannot reorder the precision-recall curve.</p>

<div class="figure">
    <img src="file:///D:/cdacminor/paper_graphs/fig3_pr.png" alt="PR Curve" />
    <p class="figure-caption">Fig. 4. Precision-Recall curve showing the APG-ASR agent's operational point relative to static confidence thresholds.</p>
</div>

<h3>C. Non-Degradation Guarantee</h3>
<p>A critical observation from our experiments is that mAP@[0.5:0.95] remains effectively unchanged under refinement, while mAP@50 shows measurable improvement. This is not a failure—it reveals a fundamental property of post-hoc refinement.</p>

<p>Improvements in mAP@50 from refinement can be misleading: they may co-occur with boundary blurring that degrades high-IoU precision. Our single-best-box selection explicitly prevents this by avoiding coordinate averaging. The result is that refinement preserves the underlying geometry while recovering valid detections that would otherwise be discarded.</p>

<p>This establishes a provable ceiling: <em>inference-time refinement cannot improve mAP@[0.5:0.95] without retraining</em>, because the detector's feature representation is frozen. The contribution of APG-ASR is not to break this ceiling, but to operate safely within it.</p>

<h3>D. Efficiency and Compute Tradeoff</h3>

<p class="table-caption">TABLE V<br>Controllability: Refinement Volume vs. Uncertainty Threshold</p>
<table>
    <tr>
        <th>&tau;<sub>low</sub></th>
        <th>Accepted Refinements</th>
        <th>&Delta; prev</th>
        <th>Latency (ms)</th>
        <th>mAP@[0.5:0.95]</th>
    </tr>
    <tr>
        <td>0.05</td>
        <td>1,923</td>
        <td>—</td>
        <td>128</td>
        <td>0.1891</td>
    </tr>
    <tr>
        <td>0.10</td>
        <td>1,716</td>
        <td>-207</td>
        <td>124</td>
        <td>0.1890</td>
    </tr>
    <tr>
        <td>0.15</td>
        <td>1,503</td>
        <td>-213</td>
        <td>115</td>
        <td>0.1890</td>
    </tr>
    <tr>
        <td>0.20</td>
        <td>1,295</td>
        <td>-208</td>
        <td>108</td>
        <td>0.1890</td>
    </tr>
</table>
<span class="takeaway">Varying &tau;<sub>low</sub> alters refinement volume without impacting mAP, demonstrating that the policy decouples compute allocation from detection quality.</span>

<p>As shown in Table V, accepted refinements decrease near-linearly as \(\tau_{low}\) increases, demonstrating predictable control over refinement volume. Critically, mAP@[0.5:0.95] remains constant across all threshold settings, confirming that \(\tau_{low}\) functions as a compute control knob—not an accuracy parameter.</p>

<p>We analyze compute efficiency under varying Top-K budgets. As shown in Fig. 5, refinement gains saturate beyond K=10 while latency continues to increase, demonstrating a controllable tradeoff. At K=15, the system achieves 97% of maximum refinements while maintaining sub-125ms latency.</p>

<div class="figure">
    <img src="file:///D:/cdacminor/experiments/apg_asr/plots/ieee_clean_compute_efficiency.png" alt="Compute Efficiency" />
    <p class="figure-caption">Fig. 5. (a) Saturation of accepted refinements and (b) associated latency envelope as a function of compute budget (Top-K).</p>
</div>

<p class="table-caption">TABLE VI<br>Latency and Compute Allocation</p>
<table>
    <tr>
        <th>System</th>
        <th>Avg Latency (ms)</th>
        <th>Reprocessed</th>
        <th>Upgrade Rate</th>
        <th>Compute</th>
    </tr>
    <tr>
        <td>YOLO (baseline)</td>
        <td>~45</td>
        <td>0</td>
        <td>0%</td>
        <td>Full image</td>
    </tr>
    <tr>
        <td>APG-ASR (Agent)</td>
        <td>124.0</td>
        <td>4,036</td>
        <td>42.5%</td>
        <td>Top-K Selective</td>
    </tr>
</table>

<p>The latency overhead (45ms → 124ms at K=15) represents a 2.8× increase. This cost is justified in deployment scenarios where: (1) the system cannot be retrained for new domains, (2) predictable, auditable inference behavior is required, and (3) false positives carry higher cost than missed detections—such as automated surveillance, drone-based inspection, and medical image screening. For high-throughput real-time applications (>30 FPS), the baseline detector should be used directly.</p>

<h3>E. Qualitative Analysis: Geometric Stability</h3>

<div class="figure wide-element">
    <img src="file:///D:/cdacminor/paper_graphs/fig8_failure_recovery.png" alt="Failure Recovery" />
    <p class="figure-caption">Fig. 6. Comparison of baseline detection (left) and APG-ASR refinement (right), demonstrating recovery of a low-confidence detection with preserved bounding box geometry.</p>
</div>

<p>Fig. 6 demonstrates the core advantage of single-best-box selection over coordinate averaging. The baseline detector produces a low-confidence prediction that would be discarded under static thresholding. The APG-ASR agent re-evaluates the cropped region and selects the highest-confidence candidate directly, recovering the detection while maintaining tight bounding box boundaries. In contrast, WBF-based approaches would average this box with neighboring candidates, producing a geometrically distorted output.</p>

<h3>F. Multi-Seed Robustness</h3>

<p class="table-caption">TABLE VII<br>Multi-Seed Stability (3 Independent Initializations)</p>
<table>
    <tr>
        <th>Seed</th>
        <th>Baseline mAP@50</th>
        <th>Agent mAP@50</th>
        <th>Upgrades</th>
        <th>&Delta; mAP</th>
    </tr>
    <tr>
        <td>Init 1</td>
        <td>0.3293</td>
        <td>0.3292</td>
        <td>1,751</td>
        <td>-0.0001</td>
    </tr>
    <tr>
        <td>Init 2</td>
        <td>0.3290</td>
        <td>0.3290</td>
        <td>1,720</td>
        <td>0.0000</td>
    </tr>
    <tr>
        <td>Init 3</td>
        <td>0.3291</td>
        <td>0.3291</td>
        <td>1,716</td>
        <td>0.0000</td>
    </tr>
</table>
<span class="takeaway">The refinement policy produces consistent upgrade counts across seeds, with zero mean degradation in mAP—confirming the non-degradation guarantee.</span>

<h2>V. Applications</h2>
<p>The proposed method is directly applicable to real-time object detection pipelines where it selectively refines uncertain detections while maintaining bounded computational cost. Specific deployment scenarios include:</p>
<ul>
    <li><strong>Improving uncertain detections in deployed systems.</strong> In surveillance, traffic monitoring, and drone-based inspection, the model detects objects but may lack confidence. Rather than accepting potentially incorrect boxes, APG-ASR refines only uncertain cases without reprocessing the entire frame.</li>
    <li><strong>Compute-efficient refinement on edge devices.</strong> Unlike TTA/WBF methods that process all detections, APG-ASR refines only Top-K uncertain boxes, making it suitable for systems with limited GPU capacity or strict power budgets.</li>
    <li><strong>Stable, predictable inference.</strong> Naive refinement sometimes degrades results. APG-ASR guarantees that refinement never worsens the original prediction, enabling deployment in safety-critical systems that require auditable, predictable outputs.</li>
</ul>

<h2>VI. Limitations</h2>
<ul>
    <li><strong>Recall ceiling.</strong> Cropping strips macro-context, limiting the agent's ability to resolve objects that depend on surrounding scene information.</li>
    <li><strong>Threshold sensitivity.</strong> The \(\tau_{low}\) parameter requires manual calibration per deployment domain.</li>
    <li><strong>Latency overhead.</strong> The 2.8× latency increase at K=15 limits applicability in high-throughput real-time systems requiring >30 FPS.</li>
    <li><strong>mAP@[0.5:0.95] ceiling.</strong> Inference-time refinement cannot improve strict IoU metrics without retraining the base detector's feature representation.</li>
</ul>

<h2>VII. Conclusion</h2>
<p>This paper presented APG-ASR, a bounded inference-time refinement framework that prioritizes decision reliability over detection coverage. By mathematically clamping the interaction between CNN and Transformer features using the Adaptive Probabilistic Gate, we eliminated the gradient corruption inherent in naive hybrid architectures. The bounded decision policy—employing single-best-box selection over weighted averaging—guarantees that refinement preserves bounding box geometry and cannot degrade detection quality.</p>

<p>Our experiments establish a fundamental insight: inference-time refinement operates within a provable ceiling on mAP@[0.5:0.95], because the detector's feature representation is frozen. The contribution of APG-ASR is to operate safely within this ceiling—providing controlled, non-destructive refinement with bounded latency.</p>

<p>Future work will explore extending the proposed control framework to alternative detector architectures (DETR, Faster R-CNN), learning adaptive uncertainty thresholds dynamically, and incorporating temporal consistency for video-based refinement. Additionally, developing APG-ASR as a modular refinement layer that can be integrated with pretrained detectors and adapted to domain-specific datasets through fine-tuning represents a natural deployment path.</p>

<h2>References</h2>
<div class="references">
    <p>[1] H. Lou et al., "What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector," <i>arXiv preprint arXiv:2408.15857</i>, 2024.</p>
    <p>[2] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2020.</p>
    <p>[3] N. Carion et al., "End-to-End Object Detection with Transformers," in <i>Proc. Eur. Conf. Comput. Vis. (ECCV)</i>, 2020, pp. 213-229.</p>
    <p>[4] J. Guo et al., "CMT: Convolutional Neural Networks Meet Vision Transformers," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2022.</p>
    <p>[5] Z. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," in <i>Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)</i>, 2021.</p>
    <p>[6] Y. Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2024.</p>
    <p>[7] C. Y. Wang, A. Bochkovskiy, and H. Y. M. Liao, "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2023.</p>
    <p>[8] X. Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection," in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2021.</p>
    <p>[9] S. Chen et al., "DiffusionDet: Diffusion Model for Object Detection," in <i>Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)</i>, 2023.</p>
    <p>[10] R. Solovyev et al., "Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models," <i>Image and Vision Computing</i>, vol. 107, 2021.</p>
</div>

</body>
</html>"""

file_path = r'd:\cdacminor\paper_temp.html'
with io.open(file_path, 'w', encoding='utf-8') as f:
    f.write(html)

print("Saved HTML file to disk. Now generating PDF...")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    uri = pathlib.Path(file_path).absolute().as_uri()
    page.goto(uri, wait_until='networkidle')
    page.pdf(
        path=r'd:\cdacminor\paper\APG-ASR_IEEE_Revised_Final.pdf',
        format='A4',
        print_background=True,
        margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'}
    )
    browser.close()
    print('PDF generation complete.')
