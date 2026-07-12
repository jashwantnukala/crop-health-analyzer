# 10-Month Research Timeline — Image-Based Crop Disease Detection
### Turmeric & Ridge Gourd | TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This timeline sequences the work exactly as it is architected in the technical architecture, model-comparison, and deployment-strategy documents already prepared — data pipeline first, then preprocessing, then parallel model training, then comparison, then optimization/compression, then explainability, then deployment engineering, then field validation and handover. No stage is scheduled before its documented prerequisite is complete. Each month is broken into the nine dimensions requested; risks are project-specific (tied to what could actually go wrong at that stage, not generic "delays may occur" language), and every deliverable is something a reviewer can independently check exists.

---

## Month 1 — Foundation: Literature, Crop Scoping, Data Ingestion

| Dimension | Details |
|---|---|
| **Objectives** | Finalize problem scope for both crops; establish the data pipeline as the single source of truth before any modeling begins. |
| **Tasks** | Complete literature review synthesis (already drafted, finalize with faculty mentor sign-off); confirm turmeric/ridge gourd disease taxonomy with TIH-IoT domain experts; receive and ingest TIH-IoT labelled dataset; run checksum/deduplication and schema validation (Stage 1 of technical architecture). |
| **Expected Deliverables** | Finalized literature review document; confirmed disease-class taxonomy per crop (signed off by TIH-IoT/faculty mentor); versioned, checksummed dataset manifest (v0); class-distribution audit report. |
| **Risk** | Dataset delivered later than expected, or with incomplete/inconsistent metadata (missing capture-source or timestamp fields), which would delay the group-aware train/val/test split downstream. |
| **Mitigation** | Request dataset delivery in Week 1 with an explicit CFP-agreed deadline; build the ingestion/validation pipeline against a small public proxy dataset (e.g., PlantVillage subset) in parallel so pipeline code is ready and only needs re-pointing once the real dataset arrives, rather than blocking all Month-1 work on delivery timing. |
| **Testing** | Unit tests on the ingestion pipeline (checksum correctness, schema-rejection on malformed records) using synthetic corrupted samples. |
| **Documentation** | Dataset datasheet (v0): collection methodology, class distribution, known limitations. |
| **Validation** | Manual spot-check of 50 randomly sampled ingested records against raw source images to confirm no metadata corruption during ingestion. |
| **Deployment** | None — infrastructure-only month. DVC repository and experiment-tracking (MLflow/W&B) initialized so every subsequent stage is versioned from day one. |

---

## Month 2 — Preprocessing, Augmentation, Stratified Splitting

| Dimension | Details |
|---|---|
| **Objectives** | Convert the raw, heterogeneous field-collected dataset into a model-ready, leakage-free, class-balance-aware training asset. |
| **Tasks** | Implement resolution standardization, colour normalization (dataset-computed, not ImageNet defaults), ROI/leaf-segmentation preprocessing; implement group-aware stratified 70/15/15 split; design and implement the augmentation policy for the train split; benchmark class-imbalance mitigation strategies (class-weighted loss vs. oversampling) empirically on a small pilot model. |
| **Expected Deliverables** | Finalized preprocessing pipeline (versioned, tested); finalized train/val/test manifest (v1) with documented split methodology; augmentation policy document with sample before/after visualizations. |
| **Risk** | Severe class imbalance (a known risk flagged in the model-comparison document) turns out worse than literature-typical ranges for one or both crops, which could make some minority disease classes practically untrainable with the available samples. |
| **Mitigation** | The class-distribution audit from Month 1 is used decisively here, not just descriptively — if a class falls below a viable-sample threshold (to be set with faculty mentor input, e.g. <30 images), flag it to TIH-IoT immediately for possible supplementary collection, rather than discovering the problem after Month 3–4 training runs have already been spent on it. |
| **Testing** | Leakage test: confirm zero image-hash overlap between train/val/test splits; augmentation sanity check (visual inspection that augmented images remain diagnostically valid, not corrupted beyond recognition). |
| **Documentation** | Preprocessing and augmentation methodology section of the technical report (incremental, not written at the end). |
| **Validation** | Compare pre- and post-ROI-segmentation classification accuracy on a small pilot CNN to confirm the segmentation step measurably helps before committing it to the full pipeline. |
| **Deployment** | None. Preprocessing module is packaged as the first version of `inference_core` (Section 2 of deployment strategy) so training-time and future serving-time preprocessing are provably identical from this point onward. |

---

## Month 3 — Baseline & Lightweight Model Training (Candidates A, C1)

| Dimension | Details |
|---|---|
| **Objectives** | Establish the traditional-ML and lightweight-CNN baselines that anchor the rest of the comparison. |
| **Tasks** | Train Category A traditional-ML baseline (hand-crafted features + classical classifier); train MobileNetV3 (C1) via transfer learning — the deployment-feasibility anchor identified in the model-comparison document; establish the evaluation harness (accuracy, F1 per class, confusion matrices) to be reused identically for every later model. |
| **Expected Deliverables** | Trained and checkpointed A and C1 models; first entries in the model leaderboard; reusable evaluation harness (code + report template). |
| **Risk** | MobileNetV3 transfer learning underperforms expectations on field-condition images specifically (the literature-documented lab-to-field accuracy drop), which could be mistaken for a modeling bug rather than the expected domain-shift effect. |
| **Mitigation** | Evaluate on both the lab-condition-like validation split and a held-out field-condition subset separately from the start (not only at Stage 6b later), so a lab/field accuracy gap is correctly attributed to domain shift immediately, with the ROI-segmentation and colour-normalization choices from Month 2 as the first lever to revisit, not model architecture. |
| **Testing** | k-fold cross-validation on the lab-condition split; regression test confirming evaluation harness produces identical metrics on repeated runs with a fixed seed. |
| **Documentation** | Model cards for A and C1 (architecture, hyperparameters, training data version, metrics). |
| **Validation** | Confusion-matrix review with faculty mentor to sanity-check that per-class errors are agriculturally plausible (e.g., confusion between visually similar disease classes), not random. |
| **Deployment** | C1 (MobileNetV3) exported to ONNX as a first end-to-end deployment-pipeline test (Section 4 of deployment strategy) — validates the export path early, before it becomes a bottleneck near the deployment milestone. |

---

## Month 4 — Deeper CNN & Transformer Candidates (C2/EfficientNet-B0, D1/ViT-B/16)

| Dimension | Details |
|---|---|
| **Objectives** | Complete the CNN and transformer arms of the model comparison to answer the project's core research question — does attention-based architecture's field-robustness advantage hold for these two under-studied crops. |
| **Tasks** | Train EfficientNet-B0 (C2) via transfer learning; train ViT-B/16 (D1) via transfer learning, with careful hyperparameter tuning given ViTs' documented sensitivity to fine-tuning configuration on limited data; run the full evaluation harness (from Month 3) identically on both. |
| **Expected Deliverables** | Trained and checkpointed C2 and D1 models; updated leaderboard with four models; a first documented comparison of CNN vs. transformer field-robustness on the actual TIH-IoT dataset. |
| **Risk** | ViT-B/16 underperforms CNNs due to the literature-documented data-hunger issue, given the labelled dataset size is fixed by what TIH-IoT provides and cannot be arbitrarily grown. |
| **Mitigation** | This is treated as a legitimate research finding, not a failure to fix at any cost — if ViT underperforms, the proposal's own literature review already anticipated this as a known risk, and the mitigation is documenting *why* (data-hunger, limited fine-tuning data) rather than over-tuning to force a result, which would misrepresent the comparison. |
| **Testing** | Same k-fold + fixed-seed regression testing as Month 3, applied consistently for fair comparison. |
| **Documentation** | Model cards for C2 and D1; interim technical report section: "CNN vs. Transformer field-robustness — early findings." |
| **Validation** | Statistical significance check (paired test across k-folds) before claiming any model is meaningfully better than another — avoids over-claiming from a single accuracy-point difference. |
| **Deployment** | None new this month — deployment focus resumes once the full candidate set (through Month 5) is available for a considered selection. |

---

## Month 5 — Full Comparison, Leaderboard, Field-Validation Round 1

| Dimension | Details |
|---|---| 
| **Objectives** | Complete the mandated multi-model comparison (CFP's "train multiple models, compare" deliverable) and run the first real field-validation exercise. |
| **Tasks** | Finalize leaderboard across all trained candidates (accuracy, F1, latency, model size, deployment difficulty — reusing the comparison table structure from the model-design document); conduct first field-validation round with TIH-IoT (real, uncurated field images, not held-out lab images); compute the lab-vs-field accuracy gap explicitly per model. |
| **Expected Deliverables** | Finalized comparative leaderboard (all trained candidates); field-validation report round 1; documented recommendation for which model(s) proceed to optimization phase. |
| **Risk** | Field-validation results diverge substantially from lab validation results (the exact failure mode the group-aware splitting in Month 2 was designed to catch and reduce, but cannot eliminate entirely since field conditions are inherently more variable than any held-out split of the same dataset). |
| **Mitigation** | This round is deliberately scheduled at the project's midpoint, not the end, specifically so a large gap still leaves five months to react — via additional field-representative augmentation, revisiting the ROI-segmentation step, or reweighting which model is prioritized for optimization — rather than discovering this in Month 9 with no time to respond. |
| **Testing** | Per-class error gallery compiled from field-validation discrepancies (feeds directly into Stage 6b's ongoing diagnostic tool, not a one-off report). |
| **Documentation** | Mid-project technical report (methodology + full leaderboard + field-validation findings) — the first complete report deliverable, ahead of the final month. |
| **Validation** | Faculty mentor and TIH-IoT review checkpoint — formal go/no-go on which model(s) proceed to the optimization phase, keeping the selection decision auditable rather than made unilaterally by the fellowship team. |
| **Deployment** | None — this is a decision checkpoint month, not a build month. |

---

## Month 6 — Optimization Phase: Hybrid CNN-ViT & Multi-Task PDI Head

| Dimension | Details |
|---|---|
| **Objectives** | Execute the CFP's "optimize the best-performing model" deliverable via the project's strongest novelty contribution — the hybrid CNN-Transformer architecture flagged in the literature review as under-benchmarked. |
| **Tasks** | Implement and train the hybrid CNN-ViT (E1) model, sequenced *after* the four baseline candidates per the model-comparison document's explicit staging; if TIH-IoT confirms PDI ground truth is available in the dataset, implement the multi-task PDI regression head (E2) as a secondary, near-zero-marginal-cost addition to the EfficientNet-B0 backbone. |
| **Expected Deliverables** | Trained hybrid CNN-ViT model with comparison against the four baseline candidates; PDI-head model variant (conditional on data availability) with regression evaluation metrics (MAE/RMSE on PDI estimate). |
| **Risk** | The hybrid architecture, being the least literature-validated and highest-implementation-risk candidate, fails to train stably or does not outperform the simpler baselines within the time budgeted — a real possibility explicitly acknowledged in the model-comparison document's own risk framing. |
| **Mitigation** | Because this is scheduled as a stretch/optimization-phase model *after* four working baselines already exist (per Month 3–5), a hybrid-model shortfall does not jeopardize the core deliverable — the four-model comparison remains complete and reportable on its own even if E1 underperforms or needs to be cut short; time-box the hybrid experiment to 3 weeks with a defined fallback (revert to optimizing the best baseline candidate instead) if it is not converging. |
| **Testing** | Ablation testing (CNN-only vs. attention-only vs. hybrid) to attribute any accuracy gain correctly to the hybrid design rather than to incidental hyperparameter differences. |
| **Documentation** | Model card for E1 (and E2 if pursued); ablation study write-up — this is the closest thing to a genuine novel research contribution in the proposal and is documented at the depth a reviewer would expect for that claim. |
| **Validation** | Cross-check hybrid model's field-validation accuracy gap against the same field-validation set used in Month 5, for a fair before/after comparison. |
| **Deployment** | None — E1/E2 are evaluated but not yet optimized for serving; that is the explicit focus of Month 7. |

---

## Month 7 — Compression, Quantization, Explainability Integration

| Dimension | Details |
|---|---|
| **Objectives** | Convert the selected best-performing model(s) from Months 5–6 into deployment-ready artifacts, and integrate explainability as a first-class inference output rather than an offline analysis. |
| **Tasks** | Apply knowledge distillation (heavy model → MobileNetV3-sized student, per deployment-strategy Section 3.1); apply INT8 post-training quantization with field-representative calibration data; re-evaluate every compressed/quantized variant on the *field*-validation set specifically, not just lab data; integrate Grad-CAM/SHAP explainability and confidence calibration (temperature scaling) into the inference pipeline. |
| **Expected Deliverables** | Distilled + quantized deployable model artifact with documented accuracy-retention numbers versus the FP32 heavy-model baseline; calibrated confidence outputs; integrated explainability module producing heatmap + confidence + fallback flag per prediction. |
| **Risk** | INT8 quantization's accuracy drop on field-validation data exceeds the pre-agreed 2-percentage-point threshold specified in the deployment strategy, particularly plausible if the hybrid CNN-ViT (attention layers are more quantization-sensitive) is the model being compressed. |
| **Mitigation** | The threshold-and-fallback policy is pre-committed, not decided under pressure: if PTQ exceeds the threshold, fall back to quantization-aware training (QAT) as already scoped in the deployment strategy, budgeting up to one additional week for QAT fine-tuning within this month's schedule buffer. |
| **Testing** | Side-by-side latency/accuracy benchmarking: FP32 vs. distilled vs. INT8-quantized, on representative edge hardware (a mid-range Android device, not just a development machine) to get realistic, not theoretical, numbers. |
| **Documentation** | Compression/quantization technical report section with full before/after metrics table; explainability methodology section. |
| **Validation** | Manual review (with domain expert / faculty mentor) of a sample of Grad-CAM heatmaps to confirm the model is attending to agriculturally meaningful lesion regions, not spurious background correlations — a direct trust-and-safety check before deployment. |
| **Deployment** | First deployable artifact produced: ONNX-exported, quantized model, tested (not yet shipped) on ONNX Runtime Mobile. |

---

## Month 8 — Serving Infrastructure: API, Docker, Desktop App

| Dimension | Details |
|---|---|
| **Objectives** | Build the serving infrastructure specified in the deployment strategy — REST API, containerization, and the desktop application — around the finalized model artifact from Month 7. |
| **Tasks** | Implement the FastAPI service (`/predict`, `/predict/batch`, `/health`, `/models` endpoints); containerize as two Docker images (CPU/ONNX Runtime and GPU/TensorRT variants); implement the Python/PyQt desktop application reusing the shared `inference_core` module; implement structured logging (Section 8 of deployment strategy) across both surfaces from the start, not retrofitted later. |
| **Expected Deliverables** | Working REST API (locally deployable, documented via auto-generated OpenAPI schema); two Docker images, tested; functional desktop application with offline inference capability. |
| **Risk** | Integration friction between the ONNX-exported model and the FastAPI/desktop serving code — e.g., input-tensor shape mismatches, or the explainability module's added latency making batch requests unacceptably slow. |
| **Mitigation** | Because `inference_core` was established as a shared module as early as Month 2 and stress-tested with the C1 ONNX export in Month 3, most integration risk is front-loaded and already surfaced by this point rather than discovered fresh in Month 8; explainability latency is addressed by the confidence-threshold gating already specified in the deployment strategy (Grad-CAM computed only above a threshold, not on every request). |
| **Testing** | API load testing (concurrent request handling); Docker image build reproducibility test (clean-environment rebuild); desktop app testing on a lower-spec reference machine matching the target field-office hardware profile. |
| **Documentation** | API documentation (auto-generated + supplementary usage guide); desktop app installation/user guide — early draft of the SOPs the CFP requires. |
| **Validation** | End-to-end validation: image in → correct structured prediction out, across both API and desktop surfaces, cross-checked against the standalone model evaluation from Month 7 to confirm no accuracy regression was introduced by the serving layer itself. |
| **Deployment** | API and desktop app deployed to a staging environment (not yet production/TIH-facing) for internal testing. |

---

## Month 9 — Edge Deployment, Monitoring, Cloud Staging, Field-Validation Round 2

| Dimension | Details |
|---|---|
| **Objectives** | Complete the edge (on-device) deployment path, stand up the monitoring/logging backbone across all surfaces, and run the second, deployment-realistic field-validation round. |
| **Tasks** | Integrate the quantized model into an ONNX Runtime Mobile test harness on real Android hardware; implement the monitoring stack (Prometheus/Grafana or equivalent, drift/confidence tracking, expert-disagreement logging); stage the cloud deployment (Docker images deployed to actual cloud/IIT-B infrastructure, coordinated with TIH-IoT); run field-validation round 2 using the fully deployed pipeline (not a research-environment evaluation) with TIH-IoT field staff as real users. |
| **Expected Deliverables** | Working offline edge inference test build; live monitoring dashboards; staged cloud deployment; field-validation round 2 report, specifically comparing round 1 (Month 5, research-environment) vs. round 2 (deployment-environment) results. |
| **Risk** | Real end-user (field-staff) testing surfaces usability or reliability issues invisible in internal testing — e.g., poor performance under actual field lighting/network conditions, or confusion in the app's UI for non-technical users. |
| **Mitigation** | Round 2 is deliberately scheduled with one month of buffer remaining before final submission (Month 10), specifically to allow a fix-and-retest cycle for usability issues; issues are triaged by severity, with model-accuracy-affecting issues prioritized over UI polish given the remaining time. |
| **Testing** | Real-device offline/online-sync testing (airplane-mode toggling to confirm the offline-queue-and-sync behaviour from the deployment strategy actually works, not just in design); monitoring-alert testing (deliberately inject drifted/out-of-distribution images to confirm the drift-detection pipeline actually fires). |
| **Documentation** | Field-validation round 2 report; monitoring/SOP documentation (how to interpret dashboards, how to respond to a drift alert) — directly addresses the CFP's "future maintenance" requirement. |
| **Validation** | Direct comparison table: lab accuracy vs. round-1 field accuracy vs. round-2 deployed-system field accuracy — the single most important validation artifact in the whole project, since it is the concrete evidence of real-world readiness the CFP is ultimately judging. |
| **Deployment** | Edge (Android test build), API, and desktop app all deployed in a TIH-IoT-facing pilot capacity; cloud staging environment live. |

---

## Month 10 — Final Validation, Documentation, Handover

| Dimension | Details |
|---|---|
| **Objectives** | Close the loop on every CFP deliverable: finalize validation evidence, complete documentation/SOPs, and hand over a maintainable system to TIH-IoT. |
| **Tasks** | Address any remaining issues surfaced in Month 9's field-validation round 2; finalize the retraining-trigger policy and confirm the monitoring→data-pipeline feedback loop is functional end-to-end (a live test: inject new field-validated data, confirm it versions correctly into Stage 1's data pipeline); compile the final technical report; finalize all SOPs; prepare the reproducibility package (versioned code repo, experiment logs, environment spec). |
| **Expected Deliverables** | Final technical report (methodology, full comparative results, field-validation evidence, deployment architecture, recommendations); complete SOP set; reproducibility package; final presentation/demo for TIH-IoT and CHANAKYA reviewers. |
| **Risk** | Time pressure in the final month leads to documentation being compressed/rushed, undermining the "future maintainability" deliverable specifically — a common failure mode in fellowship projects where the demo works but the handover documentation does not survive contact with a new maintainer. |
| **Mitigation** | This risk is structurally addressed by the schedule itself, not just resolved in Month 10 — documentation has been written incrementally every month (model cards, interim technical report sections, SOP drafts from Month 8) specifically so Month 10's documentation task is *compilation and final review*, not first-draft writing under deadline pressure. |
| **Testing** | Full regression test of the entire pipeline end-to-end (data ingestion → training reproducibility → inference → deployment → monitoring) on a clean environment, by a team member who did not build that specific component, as a genuine handover-readiness test. |
| **Documentation** | Final technical report; datasheet (final version); SOPs (adding a disease class, triggering retraining, validating a new model version before deployment, interpreting confidence-threshold fallback); dataset/model licensing and IP documentation (TIH-IoT retained ownership, per CFP's IP clause). |
| **Validation** | Formal TIH-IoT / faculty mentor sign-off against the CFP's original expected-outcomes list, item by item — the final validation step is explicitly a traceability check, confirming every deliverable promised in the proposal is demonstrably present in the delivered system. |
| **Deployment** | Full system (API, desktop, edge, monitoring) handed over to TIH-IoT in a state operable by a team member other than the original fellows, per the CFP's continuity requirement; mobile app scoped and documented as the recommended Phase 2 extension. |

---

## Cross-cutting notes on schedule realism

- **Buffer placement is deliberate, not incidental.** Field-validation round 1 (Month 5) and round 2 (Month 9) are placed with, respectively, five months and one month of runway remaining — this is what allows negative findings (a model underperforming in the field, a compression technique losing too much accuracy) to be *acted on* rather than merely reported at the end.
- **Documentation is incremental throughout**, not a Month-10 activity — every month produces a specific, checkable documentation artifact, which is what makes the Month 10 "final report" a compilation task rather than a first draft under deadline pressure.
- **The hybrid/optimization model (Month 6) is explicitly sequenced after, not parallel to, the four baseline candidates** — this is a direct schedule-level expression of the model-comparison document's own staging logic, and it structurally protects the core deliverable (a complete multi-model comparison) from the highest-risk, least-proven part of the project.
- **No month assumes a resource (dataset, compute, TIH-IoT field-staff availability) that hasn't been explicitly flagged as a dependency** — where such a dependency exists (e.g., Month 1's dataset delivery, Month 9's field-staff involvement), the risk/mitigation columns say so directly rather than assuming ideal conditions.
