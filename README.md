# Image-Based Crop Disease Detection — Turmeric & Ridge Gourd

**TIH-IoT, IIT Bombay — CHANAKYA Fellowship Program 2026**

A research proposal and technical design for an AI/ML system that detects crop diseases from images, built for the [CHANAKYA Fellowship Program 2026](https://tihiitb.org/programs/hr-skill-development/cfp-2026/) run by the Technology Innovation Hub for IoT & IoE (TIH-IoT) at IIT Bombay, under India's National Mission on Interdisciplinary Cyber-Physical Systems (NM-ICPS).

This repository documents the full research-to-deployment pipeline for two crops selected from TIH-IoT's 30-crop candidate list: **Turmeric** (*Curcuma longa*) and **Ridge Gourd** (*Luffa acutangula*), selected for their contrasting disease etiology — turmeric's dominant threat is fungal rhizome rot/leaf blotch expressed through subtle foliar symptoms, while ridge gourd is dominated by whitefly-transmitted viral Yellow Mosaic Disease (ToLCNDV) causing distinct mottling and leaf distortion, with essentially no prior published image-based diagnostic work.

**Submitted by:** Mrs. Avani Anawardekar, Department of Computer Science and Engineering, IIT Bombay — July 11, 2026

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
| — | [`CHANAKYA-FellowshipProgram-ProjectProposal-2026-Turmeric-RidgeGourd.pdf`](./CHANAKYA-FellowshipProgram-ProjectProposal-2026-Turmeric-RidgeGourd.pdf) | **The final submitted proposal** — background, problem statement, SMART objectives, deliverables, milestone plan, resources/budget, and appendix, condensed from documents 1–11 into the official CFP submission format. |
| — | [`CHANAKYA-Proposal-Summary-2Slides.pptx`](./CHANAKYA-Proposal-Summary-2Slides.pptx) | 2-slide executive summary deck (per CFP's required presentation format) — problem, 5-model approach, deployment architecture, and 10-month delivery plan at a glance. |

## Technical Approach at a Glance

**Final comparison set, as submitted** (5 architectures × 2 crops):

| Model | Role | Expected Accuracy* |
|---|---|---|
| SVM / Random Forest (hand-crafted features) | Mandatory classical-ML baseline / scientific control | 80–90% |
| MobileNetV3 (transfer learning) | Edge-deployment baseline | 93–97% |
| EfficientNet-B0 (transfer learning) | Accuracy/efficiency balance, literature-comparable benchmark | 96–98.5% |
| ResNet-50 (transfer learning) | Deep-CNN ceiling | 96–98% |
| ViT-B/16 (transfer learning) | Attention-based field robustness — the core research question | 95–98%* |
| Hybrid CNN-ViT | Novel optimization target (Month 6 stretch/ablation) | 95–98.5%* |

*Literature-grounded estimates only — actual figures are measured empirically on the TIH-provided dataset, not assumed. All models are evaluated jointly on accuracy, field robustness, latency & model size, calibration, and interpretability.

**Pipeline:** Ingestion & audit → Preprocessing & augmentation → Parallel model training (two-phase transfer learning) → Comparative evaluation (macro-F1 primary, not raw accuracy) → Explainability (Grad-CAM/Grad-CAM++) → Optimization (distillation, ONNX, INT8 quantization) → Hybrid deployment (edge / API / desktop) → Field validation & continuous improvement.

**Deployment:** A single ONNX-based inference core shared across an on-device (edge, INT8-quantized, TFLite/ONNX Mobile) path for offline/low-connectivity farmer use and a FastAPI + Docker cloud API for TIH field offices and extension-worker tablets — with a mobile app explicitly scoped as future work.

### Quantitative Targets (as submitted)

- Macro-F1 above **90% (lab)** and **85% (field)**, per crop
- Under **2 percentage points** of field-accuracy loss after quantization
- On-device inference latency **under 500 ms**

### 10-Month Milestone Plan

| Month | Milestone |
|---|---|
| 1 | Data ingestion, checksumming, schema validation, class-distribution audit — versioned dataset manifest v0, TIH-IoT sign-off |
| 2 | Preprocessing, augmentation policy, leakage-free stratified split — `inference_core` v1 packaged |
| 3 | Baseline models trained: SVM/RF + MobileNetV3 — first leaderboard entries per crop |
| 4 | EfficientNet-B0 and ViT-B/16 trained — CNN vs. Transformer field-robustness comparison drafted |
| 5 | Full comparison completed; **Field-Validation Round 1** with TIH-IoT — go/no-go checkpoint |
| 6 | Hybrid CNN-ViT trained (optimization phase); ablation study completed |
| 7 | Model compression (distillation, INT8 quantization) + Grad-CAM explainability integrated — deployable ONNX artifact |
| 8 | REST API, Docker containers, offline-capable desktop app built and staged |
| 9 | Edge (Android) deployment, monitoring/drift-detection stack, **Field-Validation Round 2** |
| 10 | Final validation, technical report, SOPs, reproducibility package, handover to TIH-IoT |

### Team & Resources

A two-student core team (one CV/ML-focused with PyTorch/CNN coursework experience, one with agricultural-science grounding in crop pathology and field data collection) supervised by the faculty mentor, with periodic access to TIH-IoT domain experts for field validation. Compute: institute HPC/GPU cluster (Colab Pro/Kaggle GPU as fallback). Tooling: PyTorch, ONNX Runtime, Optuna, MLflow, Docker. This is the team's first dedicated research effort at the intersection of computer vision and agricultural disease diagnosis — no prior publications in this specific domain are claimed; TIH-IoT provides the labelled dataset but no project budget.

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
