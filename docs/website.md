# Phase H: Explainable Intelligent Detection System (Demo Blueprint)

This document is a highly detailed, non-reduced refinement of the core principles and architectural guidelines for the UI. The goal is to build a **demonstration interface** that transparently exposes the reasoning, failures, causality, and telemetry of the APG-ASR system, proving it is a runtime controllable decision engine.

---

# 🔷 1. Core Principle & Goal
You are **NOT building a full product**. You are building a **research-grade demo interface** that answers:
> “What happened inside the system and why?”
Not just:
> “Here is the output.”

Your UI must expose ALL stages of the pipeline:
1. **Input Image**: Original unaltered image.
2. **Raw Detection**: All raw noisy boxes from the base model.
3. **Agent Decisions (YOUR USP)**: Reprocessing timeline, causality scoring, and system intelligence.
4. **Final Output**: Clean, mathematically verified, color-coded detections.
5. **Ground Truth Validation**: Explicit correctness tracking against standard annotations.

---

# 🔷 2. Architecture (Industry Standard)
**FastAPI backend + Streamlit frontend**
```
[ Streamlit UI ]  ---> HTTP POST --->  [ FastAPI API ]  --->  [ APG-ASR Pipeline ]
```
The backend serves solely as an inference engine returning structured JSON data. Streamlit handles state, visualization, and interaction. UI shows insight, not raw JSON dumps.

---

# 🔷 3. Folder Structure
```
cdacminor/
│
├── scripts/
│   ├── detector.py       (Unchanged)
│   ├── agent.py          (Unchanged)
│   ├── run_pipeline.py   (Unchanged)
│
├── deploy/               (NEW FOLDER)
│   ├── api.py          ← FastAPI backend (Model Loading & Endpoints)
│   ├── app.py          ← Streamlit UI (Frontend Layout & Logic)
│   ├── utils.py        ← Drawing helpers (Bounding boxes, Colors)
```

---

# 🔷 4. Backend (FastAPI) — CORE ENGINE

## API Flow
**POST `/predict`**
Accepts: `multipart/form-data` (Image file)
Returns: JSON (No images returned).

**Output (Structured JSON Blueprint):**
```json
{
  "system_status": {
    "latency_budget": "OK",
    "reprocess_budget_used": 2,
    "reprocess_budget_max": 5,
    "compute_mode": "SAFE"
  },
  "agent_comparison": {
    "yolo_only": {"precision": 0.45, "recall": 0.32, "avg_conf": 0.31},
    "yolo_agent": {"precision": 0.60, "recall": 0.20, "avg_conf": 0.57}
  },
  "scene_analysis": {
    "objects_detected": 14,
    "overlap_density": "HIGH",
    "noise_level": "HIGH",
    "classification": "COMPLEX",
    "triggers": [">10 objects", "overlap density high", "low average confidence"]
  },
  "system_confidence": {
    "level": "MEDIUM",
    "derived_from": ["Upgrade success rate: 50%", "Failure rate: 30%", "High noise in raw detections"]
  },
  "detections": [
    {
      "id": 3,
      "class": "tie",
      "raw_conf": 0.18,
      "reprocessed_conf": 0.48,
      "iou_quality": 0.62,
      "bbox_original": [50, 60, 80, 90],
      "bbox_final": [52, 62, 85, 92],
      "stage": "Verified"
    }
  ],
  "ignored_candidates": [
    {"class": "person", "conf": 0.17, "reason": "class_skip"},
    {"class": "bottle", "conf": 0.11, "reason": "below threshold"}
  ],
  "telemetry_impact": {
    "P4_ratio": 0.12,
    "P5_ratio": 0.10,
    "mode": "NORMAL CONTEXT",
    "mode_meaning": ["CNN working well", "Transformer not dominant", "Standard reprocess strategy used"],
    "effects": ["P4 LOW → Small-object boost ENABLED"]
  },
  "candidate_scoring": [
    {
      "id": 5,
      "class": "bottle",
      "raw_conf": 0.22,
      "uncertainty": "+2.1",
      "area": "+3.5",
      "class_weight": "+2.0",
      "total_score": 7.6,
      "decision": "SELECTED"
    }
  ],
  "failure_analysis": {
    "CRITICAL": [
      {"msg": "ID#5: bottle → lost_detection", "cause": ["crop removed context", "object too small"]}
    ],
    "MINOR": [
      {"msg": "ID#2: car → confidence_drop", "cause": ["low initial signal"]}
    ]
  },
  "narrative_explanation": "This scene is complex due to multiple overlapping persons. The model confidently detects large objects. Small objects required reprocessing. Agent improved 1 detection but failed on 1 due to crop loss. Final output prioritizes precision over recall."
}
```

---

# 🔷 5. Frontend (Streamlit) — WHAT PEOPLE SEE

## 5.1 Layout & Visual Design

**Top Bar:**
- `[ Upload Image ]`
- **Toggle Agent ON/OFF (MANDATORY)**: Live toggle to swap data sources.
- **Toggle: Show Ground Truth**: Draws GT boxes in WHITE against Final boxes to prove accuracy over confidence.

**Main Screen (2 Columns):**
```
| Original Image / Show Raw | Processed Image / Show Final |
```
- **Feature 1: Bounding Box Drift Visualization**: For upgraded objects, draw the old box (dashed) and the new box (solid) showing explicit coordinate shifts. Color intensity = IoU quality.
- **Feature 2: Replay Mode (KILLER FEATURE)**: A button `▶ Replay Decisions`. The system visually steps through: 1) Raw detections 2) Selection 3) Reprocess 4) Filtering.
- **Feature 3: “Explain This Image” Button (ULTIMATE FEATURE)**: Pops out a human-readable text paragraph synthesizing the system's reasoning logic (`narrative_explanation`).

## 5.2 Normalized Terminology & Color Visualization
| Stage | Name | Color |
|---|---|---|
| Ground Truth | **Ground Truth** | ⬜ WHITE |
| High conf | **Trusted** | 🟩 GREEN |
| Medium conf improved | **Verified** | 🟦 BLUE |
| Weak kept | **Retained** | 🟧 ORANGE |
| Removed | **Rejected** | 🟥 RED / Hidden |
| Raw Detections | **Noise** | 🟥 RED (Overlay only) |

## 5.3 Dashboards (The "Gold" Features)

**1. Agent OFF vs ON Metrics Panel (Live Numeric Comparison)**
```
YOLO ONLY:               YOLO + AGENT:
Precision: 0.45          Precision: 0.60
Recall: 0.32             Recall: 0.20
Avg Conf: 0.31           Avg Conf: 0.57
```
let
**2. Real-Time Constraint Awareness & Compute**
```
SYSTEM STATUS
- Latency budget: OK
- Reprocess budget: 2 / 5 used (NOT EXHAUSTED)
- Compute mode: SAFE
- Total extra inference cost: +117 ms
(⚠️ LATENCY LIMIT APPROACHING → Stopping further reprocessing)
```

**3. Confidence Distribution Graph**
A simple histogram (e.g. using Streamlit bar charts) showing Raw vs Final confidence. Proves: "We removed low-confidence noise".

**4. Telemetry Impact & System Mode Explanation**
```
TELEMETRY IMPACT
P4: ██████░░░░ 12% (LOW)
P5: █████░░░░░ 10% (NORMAL)

System Mode: NORMAL CONTEXT
Meaning:
- CNN working well
- Transformer not dominant
- Balanced feature fusion
→ Standard reprocess strategy used

Effects:
- P4 LOW → Small-object boost ENABLED
```

**5. Candidate Scoring Breakdown (Causality)**
```
CANDIDATE SELECTION LOGIC
ID#5: bottle (0.22)
- uncertainty: HIGH (+2.1)
- area: SMALL (+3.5)
- class_weight: HIGH (+2.0)
→ TOTAL SCORE: 7.6 → SELECTED
```

**6. What the Agent Ignored Panel**
```
IGNORED CANDIDATES:
- person (0.17) → class_skip
- bottle (0.11) → below threshold
```

**7. Scene Complexity & System Confidence**
```
SCENE ANALYSIS
Objects detected: 14
Overlap density: HIGH
Noise level: HIGH
→ Scene classified as: COMPLEX
Triggered because: >10 objects, overlap density high.

SYSTEM CONFIDENCE
Overall Confidence: MEDIUM
Derived from:
- Upgrade success rate: 50%
- Failure rate: 30%
- High noise in raw detections
```

**8. Object-Level Tracking & Bounding Box Drift (IoU)**
```
ID#3: tie
- raw: 0.18
- reprocessed: 0.48
- IoU quality: 0.62 → Verified (Box shifted)

ID#5: bottle
- raw: 0.22
- reprocessed: 0.00
- IoU quality: 0.32 → Rejected
```

**9. Failure Cause Attribution**
```
FAILURE ANALYSIS
CRITICAL:
- ID#5: bottle → lost_detection
  CAUSE: crop removed context, object too small

MINOR:
- ID#2: car → confidence_drop
  CAUSE: low initial signal
```

---

# 🔷 6. Strict Build Execution Plan
No more redesign. The execution follows this strict 6-phase plan.

## PHASE 1 — FASTAPI BACKEND (CORE ENGINE)
**Goal:** Make the pipeline callable via HTTP.
- **Step 1**: Create `deploy/api.py`.
- **Step 2**: Load system ONCE outside endpoint (`Detector()`, `Agent()`).
- **Step 3**: Create `/predict` endpoint that returns strict JSON (no images).
- **Step 4**: Run server with uvicorn.
- **Step 5**: Test manually via cURL.
*Done when: API returns JSON without crashing.*

## PHASE 2 — STREAMLIT BASIC UI
**Goal:** Upload image → get response → show output.
- **Step 1**: Create `deploy/app.py`.
- **Step 2**: Implement basic Streamlit UI to upload image and POST to `127.0.0.1:8000/predict`.
- **Step 3**: Print JSON to screen.
*Done when: Image uploads and JSON prints.*

## PHASE 3 — IMAGE VISUALIZATION
**Goal:** Make detections visible (not raw JSON).
- **Step 1**: Create `deploy/utils.py` with `draw_boxes()` helper.
- **Step 2**: Show Original (Raw boxes in RED) vs Final (Processed boxes in GREEN) in `app.py`.
*Done when: Raw and Final boxes are visible and the difference is obvious.*

## PHASE 4 — CORE INTELLIGENCE DISPLAY
**Goal:** Show reasoning, not just detection.
- Add 4 MUST panels: 
  1) Agent ON vs OFF comparison
  2) Decision Trace
  3) Telemetry
  4) Failures
*Done when: You can explain system behavior from UI alone.*

## PHASE 5 — ADVANCED VISUAL LOGIC
**Goal:** Make system look intelligent and analytical.
- Add Confidence Histogram.
- Add Ignored candidates panel.
- Add Scene analysis.
- Add System confidence.
*Done when: UI shows *why*, not just *what*.*

## PHASE 6 — HIGH-IMPACT FEATURES
**Goal:** Push demo to top level.
- Agent Toggle checkbox (sends flag to API).
- Ground truth overlay.
- Bounding box drift (dashed to solid).
- "Explain This Image" Button.
- Replay Mode.
*Done when: Anyone can understand the pipeline visually.*

# 🧠 FINAL RULES
1. If feature doesn't improve clarity → remove it.
2. Never hide failures → show them clearly.
3. Always show BEFORE vs AFTER.
4. UI must explain decisions, not just display output.

# 🔷 7. Final Truth
This is about proving: **“My system is not just detecting. It is reasoning.”**
If someone opens your demo and says: “I understand exactly what your system is doing” → You succeeded.
If they say: “Cool boxes” → You failed.
