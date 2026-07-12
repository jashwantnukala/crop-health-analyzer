# Technical Risk Register — Image-Based Crop Disease Detection
### Turmeric & Ridge Gourd | TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This register consolidates every technical risk already flagged individually across the literature review, technical architecture, model-comparison, deployment-strategy, and timeline documents into one traceable table, plus additional risks not yet made explicit elsewhere. Each risk is stated as a specific failure mode (not a generic category), rated for likelihood/impact given this project's actual scope, and paired with a mitigation that is already load-bearing in the design — not a new promise bolted on for this document. Where a risk cannot be fully eliminated, that is stated honestly, since an IIT reviewer will discount a proposal that claims zero residual risk more than one that reasons about it.

**Likelihood/Impact scale:** Low / Medium / High, assessed specifically for a 10-month UG fellowship with a TIH-IoT-provided dataset — not generic ML-project risk ratings.

---

## 1. Dataset Issues

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1.1 | Dataset delivery from TIH-IoT is delayed beyond Month 1, compressing every downstream stage. | Medium | High | Ingestion/validation pipeline is built and tested against a public proxy dataset (PlantVillage subset) in parallel, so only re-pointing to the real dataset is needed on delivery — Month 1 work is not blocked on delivery timing (already scheduled this way in the timeline document). |
| 1.2 | Incomplete or inconsistent metadata (missing capture-source, timestamp, or collection-event ID), which breaks the group-aware split and risks train/test leakage. | Medium | High | Strict schema validation at ingestion (Stage 1 of technical architecture) quarantines non-conforming records immediately rather than silently accepting them; any record missing group-identifying metadata is flagged to TIH-IoT within Month 1, not discovered mid-training. |
| 1.3 | Duplicate or near-duplicate images across the dataset (same plant photographed multiple times), causing inflated validation accuracy if not caught before splitting. | Medium | High | SHA-256 checksumming plus a near-duplicate detection pass (perceptual hashing) at ingestion; group-aware stratified splitting ensures any near-duplicates that do exist stay within one split rather than crossing train/test. |
| 1.4 | Label noise — TIH-IoT's ground-truth labels contain errors (mislabeled disease class, ambiguous/co-occurring symptoms), which is common in agricultural datasets collected in the field rather than under lab supervision. | Medium | Medium | A held-out subset is manually cross-checked against domain-expert judgement (faculty mentor / TIH-IoT agronomist) early, to estimate a label-noise rate; if material, label-smoothing or confident-learning-style noise-robust training techniques are applied, and the noise rate is disclosed in the final report rather than treated as if labels are ground truth by definition. |
| 1.5 | Dataset size is smaller than expected for one or both crops, insufficient for data-hungry architectures (notably ViT-B/16). | Medium | Medium | Transfer learning (not from-scratch training) is the default for every candidate specifically to reduce data requirements; if ViT-B/16 underperforms due to data scarcity, this is documented as a legitimate finding (per the timeline document's Month 4 risk framing), not treated as a failure requiring endless re-tuning. |
| 1.6 | Domain mismatch between TIH-IoT's collection conditions and true field deployment conditions (e.g., dataset collected under semi-controlled conditions but deployment targets fully uncontrolled smartphone captures). | Medium | High | Two-stage field-validation (Month 5 and Month 9 rounds) specifically tests on deployment-realistic images, not just held-out samples from the same collection batch — this is the direct check for this exact risk. |

---

## 2. Class Imbalance

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 2.1 | Majority disease classes dominate (literature-typical 20–30× imbalance ratio), causing the model to trivially predict the majority class and report misleadingly high overall accuracy. | High | High | Class-distribution audit performed immediately after ingestion (Month 1); per-class F1/recall reported alongside overall accuracy in every evaluation, so a majority-class-collapse model cannot pass evaluation on aggregate accuracy alone. |
| 2.2 | One or more minority classes have too few samples to train reliably at all (below a viable-sample threshold). | Medium | High | Explicit threshold check (proposed: <30 images) flagged to TIH-IoT in Month 2 for possible supplementary collection; if unresolved, the affected class is documented as an explicit scope limitation rather than silently dropped or misreported as covered. |
| 2.3 | Naively random oversampling of minority classes causes overfitting to a small number of repeated minority-class images. | Medium | Medium | Class-weighted loss is evaluated empirically against oversampling on a pilot model (Month 2) before committing to either — the choice is evidence-based, and if oversampling is used, it is combined with augmentation (not verbatim duplication) to avoid the model memorizing exact minority-class images. |
| 2.4 | Class imbalance differs meaningfully between turmeric and ridge gourd, requiring different mitigation strategies per crop rather than one shared policy. | Medium | Low | Class-distribution audit and imbalance-mitigation choice are performed independently per crop, not assumed shared — stated explicitly as a per-crop, evidence-based decision. |

---

## 3. Overfitting

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 3.1 | High validation accuracy that does not hold on the field-validation set — the central overfitting risk for this project, distinct from classic overfitting since it can occur even with a clean train/val split if the split itself doesn't represent field diversity. | High | High | This is the primary reason field-validation (Month 5, Month 9) exists as a separate evaluation track from standard k-fold validation — a model is not considered validated on lab metrics alone anywhere in this project's methodology. |
| 3.2 | Transfer-learning fine-tuning overfits quickly on a small dataset, particularly for higher-capacity models (ResNet-50, ViT-B/16, hybrid CNN-ViT). | Medium | Medium | Standard regularization (dropout, weight decay, early stopping on validation loss) applied consistently across candidates; layer-freezing strategy (fine-tune only later layers initially) evaluated per model rather than full fine-tuning by default, since full fine-tuning is more overfitting-prone on limited data. |
| 3.3 | Aggressive data augmentation is tuned to *look* helpful on the validation split but doesn't reflect real-world image variation, creating a false sense of robustness. | Medium | Medium | Augmentation policy is sanity-checked visually (Month 2) for diagnostic plausibility, and its actual effect is measured via the lab-vs-field accuracy gap (Month 5 onward), not validation-split accuracy alone. |
| 3.4 | Hyperparameter tuning performed by repeatedly checking the same validation set risks the validation set itself being implicitly "used up" (information leakage through repeated tuning decisions). | Low | Medium | k-fold cross-validation for model selection; test set held out and touched only for final reported metrics per model, not used during iterative tuning. |

---

## 4. Hardware Failure

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 4.1 | Loss of local compute (laptop/workstation failure) during an active training run, losing unsaved progress. | Medium | Medium | Experiment tracking (MLflow/W&B, established from Month 1) checkpoints model weights and training state at regular intervals, not just at run completion; all code and configs versioned in the DVC/git repository so a run can be reproduced from a checkpoint on different hardware. |
| 4.2 | Insufficient or unavailable GPU compute for training the heavier candidates (ResNet-50, ViT-B/16, hybrid CNN-ViT) within the team's available infrastructure. | Medium | High | This is flagged explicitly as an open dependency (already noted in the deployment-strategy handoff) — GPU access (institute cluster, cloud credits, or TIH-IoT-provided compute) needs confirmation with the faculty mentor before Month 3; if unresolved, the schedule prioritizes the lighter candidates (MobileNetV3, EfficientNet-B0) first, since they are also the deployment-feasibility anchors, deferring compute-heavy candidates rather than blocking the whole project. |
| 4.3 | Target edge-deployment hardware (representative low/mid-range Android device) is not available for realistic on-device latency/accuracy testing (Month 7, Month 9). | Medium | Medium | A small, defined set of reference Android devices (spanning low-to-mid Android hardware tiers, matching the CFP's target user base) is identified and budgeted for early, rather than testing only on a high-end development phone that would give unrealistically optimistic latency numbers. |
| 4.4 | Cloud/on-premise serving infrastructure (Month 9 staging) has downtime or resource limits during the field-validation pilot, disrupting real-user testing. | Low | Medium | Docker-containerized design (deployment strategy Section 6.4) allows redeployment to alternate infrastructure with minimal reconfiguration; health-check monitoring (`/health` endpoint) surfaces downtime immediately rather than being discovered only when field staff report failures. |

---

## 5. Deployment Problems

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 5.1 | Train-serve skew — preprocessing logic implemented differently at inference time than at training time, silently degrading accuracy without throwing errors. | Medium | High | This exact risk is why `inference_core` is designed as a single shared, versioned module consumed by training, API, desktop, and edge paths alike (deployment strategy Section 2) — not implemented per-platform. |
| 5.2 | Quantization (INT8) causes an unacceptable accuracy drop on field-condition images specifically, worse than on clean lab test images. | Medium | Medium | Every compressed/quantized variant is re-evaluated on the field-validation set, not just the lab test set (deployment strategy Section 3.2); a pre-committed threshold (2 percentage points) and fallback to quantization-aware training are already scoped, so this is a planned decision path, not an emergency response. |
| 5.3 | TensorRT engine build fails or underperforms due to INT8 calibration-set mismatch (a documented, common TensorRT failure mode). | Medium | Low | Calibration performed using a representative field-image calibration set (deployment strategy Section 5), not random or synthetic data; TensorRT is scoped only to the GPU cloud path, so a build failure there does not block the on-device/API-CPU fallback path, which remains independently functional. |
| 5.4 | Offline/online sync logic (edge and desktop paths) fails to correctly queue and later sync predictions, causing silent data loss for farmers using the app without connectivity. | Medium | Medium | Explicit testing of the offline-queue-and-sync behaviour under real network-toggling conditions (airplane-mode test, already scheduled in Month 9), not just assumed to work from design alone. |
| 5.5 | Docker image works in the development environment but fails to build or run identically on TIH-IoT's actual target infrastructure (the classic "works on my machine" handover failure). | Medium | Medium | Clean-environment rebuild test (already scheduled in Month 8) — the image is built and run from scratch on a machine that did not build it originally, as the actual acceptance criterion, not a demo on the developer's own machine. |
| 5.6 | Desktop application's packaged size or dependency footprint is impractical for lower-spec field-office machines. | Low | Medium | Python/PyQt chosen specifically over Electron for this reason (deployment strategy Section 6.2); packaged install size is tested against the target hardware profile, not just a developer laptop. |

---

## 6. Generalization Errors

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 6.1 | Model generalizes poorly to disease presentations, lighting, or backgrounds not represented in the TIH-IoT training data (out-of-distribution field images) — the core lab-to-field generalization gap identified throughout the literature review. | High | High | ROI/leaf-segmentation preprocessing specifically targets background-clutter generalization; two rounds of field-validation (Month 5, Month 9) are the direct empirical check, with the lab-vs-field accuracy gap reported explicitly rather than only the lab number. |
| 6.2 | Model generalizes poorly across growth stages or seasons of the crop not well-represented in the dataset (e.g., trained mostly on mature-leaf images, deployed on young-plant images). | Medium | Medium | Class-distribution audit is extended to check growth-stage/seasonal metadata where available; flagged to TIH-IoT as a documented scope limitation if seasonal coverage is narrow, rather than an implicit, unstated assumption. |
| 6.3 | Confusion between visually similar disease classes (or between disease symptoms and non-disease stress factors like nutrient deficiency or drought stress, which the dataset may not distinguish). | Medium | Medium | Confusion-matrix review with faculty mentor/domain expert (already scheduled Month 3) specifically checks whether errors are agriculturally plausible; per-class error gallery from field-validation discrepancies is maintained as a standing diagnostic, not a one-time check. |
| 6.4 | Model's confidence scores are poorly calibrated — high confidence on wrong predictions, undermining the fallback/escalation logic that depends on confidence thresholds. | Medium | Medium | Temperature scaling (calibration) explicitly implemented and evaluated as part of the explainability integration (Month 7), with calibration quality checked via reliability diagrams, not assumed from raw softmax outputs. |

---

## 7. Explainability Issues

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 7.1 | Grad-CAM/SHAP heatmaps highlight spurious background regions (e.g., soil, hands, unrelated leaf areas) rather than the actual diagnostic lesion — indicating the model may be learning shortcut correlations rather than genuine disease features. | Medium | High | Manual review of heatmap samples with a domain expert (already scheduled Month 7) is a direct trust-and-safety gate before deployment, not an optional nice-to-have visualization; if shortcut learning is found, it feeds back into revisiting the ROI-segmentation and augmentation strategy, not just noted and shipped anyway. |
| 7.2 | Explainability computation (Grad-CAM) adds unacceptable latency to the inference pipeline, especially for batch requests or resource-constrained edge devices. | Medium | Medium | Confidence-threshold gating (deployment strategy Section 2) computes the heatmap only above a configurable confidence threshold, controlling the latency cost rather than computing it unconditionally on every request. |
| 7.3 | Explainability outputs are technically correct but not interpretable to the actual end user (a farmer or extension worker without ML background) — a heatmap alone doesn't communicate "why" in an actionable way. | Medium | Medium | Field-validation rounds include direct feedback from TIH-IoT field staff on the explainability presentation (not just model accuracy), so interpretability is validated with the real target audience rather than assumed sufficient because it's technically implemented. |
| 7.4 | Attention-based visualization for transformer/hybrid models (ViT-B/16, hybrid CNN-ViT) is less mature and harder to validate than Grad-CAM for CNNs, given the different underlying mechanism. | Medium | Low | Attention-rollout or equivalent transformer-specific visualization technique used for D1/E1 rather than forcing Grad-CAM onto an architecture it wasn't designed for; if the shipped model is ultimately the distilled MobileNetV3/EfficientNet-B0 (the deployment-strategy's default recommendation), this risk is largely moot for the production path and only matters for the research-comparison reporting. |

---

## 8. Timeline Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 8.1 | Dataset or field-staff dependencies (Month 1 delivery, Month 5/9 field-validation rounds) slip due to factors outside the fellowship team's control. | Medium | High | Already addressed structurally in the timeline document — parallel pipeline-building against a proxy dataset (Month 1), and buffer placement (5 months after round 1, 1 month after round 2) specifically to absorb slippage without collapsing the whole schedule. |
| 8.2 | The hybrid CNN-ViT optimization phase (Month 6) — the highest-implementation-risk, least literature-validated component — runs over its allocated time and threatens later months. | Medium | Medium | Explicitly time-boxed to 3 weeks with a pre-defined fallback (revert to optimizing the best baseline candidate) already specified in the timeline document — this risk is pre-committed to a resolution, not open-ended. |
| 8.3 | Compression/quantization (Month 7) requires an unplanned QAT fallback cycle, consuming schedule buffer intended for other purposes. | Medium | Medium | QAT fallback is pre-budgeted as up to one additional week within Month 7 specifically (timeline document), rather than being an unbudgeted surprise if PTQ's accuracy drop exceeds threshold. |
| 8.4 | Final-month (Month 10) documentation and handover work is compressed if earlier months run over. | Medium | High | Structurally mitigated by incremental documentation practice enforced from Month 1 onward (model cards, interim report sections, SOP drafts by Month 8) — Month 10 is scoped as compilation and final review, not first-draft writing, so it degrades gracefully under time pressure rather than failing outright. |
| 8.5 | Team capacity (a small UG fellowship team) is insufficient to execute all five model candidates, full deployment stack, and two field-validation rounds within 10 months. | Medium | High | Explicit, pre-agreed deprioritization order already established in the model-comparison document (ResNet-50 and the from-scratch CNN baseline are the first candidates to cut if time is short); the hybrid model is a stretch goal, not a core commitment — the schedule is designed so cutting the lowest-priority items still leaves a complete, defensible core deliverable (four models, full comparison, one deployment path, two field-validation rounds). |

---

## Summary: risk-management philosophy

Every mitigation in this register is either (a) already structurally built into the architecture/timeline documents already prepared for this proposal — a pre-committed decision path, not a promise made under future pressure — or (b) an explicit, honest scope limitation to be disclosed rather than hidden if it cannot be fully resolved (e.g., an unviable minority disease class, a ViT underperforming due to data scarcity). This distinction matters to a reviewer: a proposal that pre-commits to specific thresholds, fallback plans, and disclosure practices for its own risks demonstrates the kind of engineering maturity the CHANAKYA fellowship is evaluating, as opposed to a proposal that asserts risks are unlikely without a concrete plan for when they occur anyway.
