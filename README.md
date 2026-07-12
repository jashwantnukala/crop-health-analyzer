# Image-Based Crop Disease Detection — Turmeric & Ridge Gourd

**TIH-IoT, IIT Bombay — CHANAKYA Fellowship Program 2026**

A research proposal and technical design for an AI/ML system that detects crop diseases from images, built for the [CHANAKYA Fellowship Program 2026](https://tihiitb.org/programs/hr-skill-development/cfp-2026/) run by the Technology Innovation Hub for IoT & IoE (TIH-IoT) at IIT Bombay, under India's National Mission on Interdisciplinary Cyber-Physical Systems (NM-ICPS).

This repository documents the full research-to-deployment pipeline for two crops selected from TIH-IoT's 30-crop candidate list: **Turmeric** (*Curcuma longa*) and **Ridge Gourd** (*Luffa acutangula*).

---

## About the Program

CHANAKYA (Comprehensive and Holistic Advancement of National Knowledge Yield and Analytics) pairs a faculty mentor and a student team (2 UG students, or 1–2 PG students) to work on an IoT/AI research project addressing a real societal problem. This project addresses the **Agriculture** application vertical: image processing for crop disease identification.

- **Funding body:** TIH-IoT, IIT Bombay (NM-ICPS, DST, Government of India)
- **IP ownership:** Remains with TIH-IoT, IIT Bombay
- **Fellowship duration:** UG — up to 10 months · PG — till Dec 2027 or end of degree
- **Deliverable mandate:** 3–5 candidate models per crop, comparative evaluation, field validation, and a deployment-ready inference pipeline

## Project Goal

Build, compare, and deploy AI/ML models that identify diseases and pests in turmeric and ridge gourd from ordinary smartphone images — robust enough for real field conditions in rural India, not just clean lab-condition benchmarks.

## Repository Structure

Each document builds directly on the one before it — this is a single coherent pipeline, not a set of independent notes.

| # | Document | Description |
|---|---|---|
| 1 | [`1-literature-review-crop-disease-detection.md`](./1-literature-review-crop-disease-detection.md) | Grounds the project in existing research: global/Indian crop-loss impact, weaknesses of manual diagnosis, prior AI/CV solutions, research gaps, and deployment challenges. |
| 2 | [`2-crop-selection-turmeric-ridge-gourd.md`](./2-crop-selection-turmeric-ridge-gourd.md) | Nine-axis scoring framework ranking all 30 TIH-IoT candidate crops, justifying the selection of turmeric and ridge gourd. |
| 3 | [`3-technical-architecture-crop-disease-detection.md`](./3-technical-architecture-crop-disease-detection.md) | End-to-end system design — data pipeline, training, evaluation, explainability, and deployment planes — as a production ML system, not a notebook. |
| 4 | [`4-model-design-comparison.md`](./4-model-design-comparison.md) | Candidate model families (classical ML, CNNs, ViT, hybrid CNN-ViT) scored on accuracy, latency, deployability, and research value; recommends a 4-model comparison set. |
| 5 | [`5-dataset-preparation-pipeline.md`](./5-dataset-preparation-pipeline.md) | Operational spec for turning raw TIH-IoT images into a leakage-free, class-balance-aware training set. |
| 6 | [`6-ai-training-strategy.md`](./6-ai-training-strategy.md) | Two-phase transfer-learning strategy, hyperparameters, and per-architecture training procedure. |
| 7 | [`7-evaluation-methodology.md`](./7-evaluation-methodology.md) | Predictive-quality and operational metrics, and how they're combined into a fair, defensible model comparison. |
| 8 | [`8-explainable-ai.md`](./8-explainable-ai.md) | Grad-CAM/Grad-CAM++ and complementary XAI techniques, serving farmers, domain experts, and the research team differently. |
| 9 | [`9-deployment-strategy.md`](./9-deployment-strategy.md) | Hybrid deployment architecture: ONNX-unified edge, API, and desktop serving, with monitoring, logging, and a scoped future-mobile roadmap. |
| 10 | [`10-project-timeline.md`](./10-project-timeline.md) | Month-by-month 10-month plan — objectives, tasks, deliverables, risks, and validation for each stage. |
| 11 | [`11-risk-register.md`](./11-risk-register.md) | Consolidated technical risk register spanning dataset, class-imbalance, modeling, and deployment risks, each with a concrete mitigation. |
| — | [`CHANAKYA-FellowshipProgram-ProjectProposalFormat-2026.docx`](./CHANAKYA-FellowshipProgram-ProjectProposalFormat-2026.docx) | Official TIH-IoT proposal template used to submit this project. |

## Technical Approach at a Glance

**Recommended model comparison set** (4 of the CFP-required 3–5 models):

| Model | Role | Expected Accuracy* |
|---|---|---|
| SVM / Random Forest (hand-crafted features) | Mandatory classical-ML baseline / scientific control | 80–90% |
| MobileNetV3 (transfer learning) | Deployment-feasibility anchor (edge/on-device target) | 93–97% |
| EfficientNet-B0 (transfer learning) | Literature-comparable accuracy benchmark | 96–98.5% |
| ViT-B/16 (transfer learning) | Core research question — does attention-based field robustness hold for these crops | 95–98%* |

*Literature-grounded estimates only — actual figures are measured empirically on the TIH-provided dataset, not assumed.

**Pipeline:** Ingestion & audit → Preprocessing & augmentation → Parallel model training (two-phase transfer learning) → Comparative evaluation (macro-F1 primary, not raw accuracy) → Explainability (Grad-CAM/Grad-CAM++) → Optimization (ONNX, quantization, compression) → Hybrid deployment (edge / API / desktop) → Field validation & continuous improvement.

**Deployment:** A single ONNX-based inference core shared across an on-device (edge, INT8-quantized) path for offline/low-connectivity farmer use, a FastAPI + ONNX Runtime/TensorRT service for TIH field offices, and a desktop app for intermittent-connectivity settings — with a mobile app explicitly scoped as future work.

## Key Design Principles

- **Per-crop models, not one universal classifier** — disease dynamics and class imbalance differ meaningfully between turmeric and ridge gourd.
- **Macro-F1 over raw accuracy** for model selection, since disease-class imbalance can make accuracy misleadingly high.
- **Deployment feasibility as a selection criterion**, not an afterthought applied to whichever model scores highest offline.
- **Explainability as infrastructure**, not a post-hoc add-on — Grad-CAM outputs feed both the farmer-facing UI and the deployment confidence-threshold fallback.
- **Training data honestly resembles field deployment conditions** — avoiding over-sanitized datasets that don't generalize to real farmer-captured images.
- **Every risk stated with a concrete mitigation**, not asserted away — see the risk register.

## Status

This repository currently contains the **research proposal and technical design** submitted for the CHANAKYA Fellowship Program 2026 (application window closed 12 July 2026). Implementation code, datasets, and trained models will be added as the fellowship progresses.

## Program Links

- [CHANAKYA Fellowship Program 2026 — Call for Proposals](https://tihiitb.org/programs/hr-skill-development/cfp-2026/)
- [TIH-IoT, IIT Bombay](https://tihiitb.org/)
