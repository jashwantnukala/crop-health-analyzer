# Technical Architecture — Image-Based Crop Disease Detection
### Crops: Turmeric (*Curcuma longa*) & Ridge Gourd (*Luffa acutangula*)
### TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This document specifies the end-to-end system architecture, from raw labelled data as received from TIH-IoT through to a deployed, monitored inference service. It is written as a production ML system design, not a notebook pipeline — every stage has explicit inputs, outputs, failure modes, and a justification for why it exists.

---

## 0. System-level diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              OFFLINE / TRAINING PLANE                            │
│                                                                                    │
│  ┌──────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────────┐      │
│  │  TIH-IoT │──▶│  Data Pipeline │──▶│ Preprocessing│──▶│ Data Augmentation │      │
│  │  Dataset │   │ (ingest+audit) │   │  (clean/norm)│   │   (train split)  │      │
│  └──────────┘   └───────────────┘   └──────────────┘   └────────┬─────────┘      │
│                                                                   │                │
│                                                                   ▼                │
│  ┌──────────────────┐   ┌───────────────┐   ┌──────────────┐   ┌───────────────┐  │
│  │ Feature           │◀──│ Model Training│──▶│  Validation  │──▶│Hyperparameter │  │
│  │ Engineering (opt.)│   │ (5 candidates)│   │ (k-fold+field)│   │ Optimization  │  │
│  └──────────────────┘   └───────────────┘   └──────────────┘   └───────┬───────┘  │
│                                                                          │          │
│                                                                          ▼          │
│                          ┌────────────────┐   ┌──────────────────┐                 │
│                          │Model Comparison│──▶│ Explainable AI   │                 │
│                          │ (leaderboard)  │   │ (Grad-CAM/SHAP)  │                 │
│                          └────────────────┘   └────────┬─────────┘                 │
└──────────────────────────────────────────────────────────┼────────────────────────┘
                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                ONLINE / SERVING PLANE                            │
│                                                                                    │
│   ┌────────────┐    ┌───────────────────┐    ┌───────────────┐                   │
│   │ Deployment │───▶│ Monitoring &        │───▶│ Documentation │                  │
│   │ (edge+API) │    │ Feedback Loop       │    │ (SOPs/reports)│                  │
│   └────────────┘    └─────────┬───────────┘    └───────────────┘                  │
│                                │                                                   │
│                                └──────────────► feeds back into                   │
│                                                  Data Pipeline (retraining loop)   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

The loop from Monitoring back into the Data Pipeline is deliberate and required by the CFP's "continuous improvement based on new ground-truth data" deliverable — this is not a linear pipeline, it's a cycle. I'll flag this at each stage where it matters.

---

## 1. Data Pipeline

**Purpose:** Take the labelled dataset provided by TIH-IoT and turn it into a versioned, auditable, queryable data asset — not a folder of images.

**Inputs:** Raw image files + label metadata (crop, disease class, capture conditions if available) supplied by TIH-IoT for Turmeric and Ridge Gourd.

**Stages:**

1. **Ingestion & checksumming.** Every image is hashed (SHA-256) on arrival to detect duplicates and corrupted files before they enter any training run — duplicate leakage between train/val/test splits is one of the most common silent causes of inflated benchmark accuracy in the plant-disease literature (a model "memorizing" a near-duplicate image across splits, not actually generalizing).
2. **Schema validation.** Enforce a strict manifest: `image_id, crop, disease_class, capture_source (TIH/field), resolution, timestamp`. Reject or quarantine any record that doesn't conform — this catches label typos and missing metadata early rather than at training time.
3. **Dataset versioning.** Use DVC (Data Version Control) or a lightweight equivalent to snapshot every dataset revision. This matters concretely for this project because TIH-IoT explicitly expects "continuous improvement based on newly available ground-truth data" — without versioning, you cannot answer "which model was trained on which data" six months into the fellowship, which undermines the comparative-evaluation deliverable itself.
4. **Class distribution audit.** Compute per-class image counts for both crops immediately. Given known class-imbalance patterns in agricultural datasets (majority classes can be 20–30× larger than minority classes), this audit determines whether stratified sampling, class-weighted loss, or synthetic minority oversampling is needed downstream — decided by evidence, not assumed upfront.
5. **Train / validation / test split — stratified and *group-aware*.** Standard random splitting is insufficient here: if multiple images come from the same plant/leaf/field visit (common in agricultural datasets), a random split can leak near-identical images across train and test, producing misleadingly high validation accuracy that collapses under real field testing. Splits must be grouped by source plant/collection-event ID where that metadata exists, and stratified by disease class to preserve class balance across splits (suggested ratio: 70/15/15).

**Output:** A versioned, checksummed, class-balanced-aware dataset manifest — the single source of truth every downstream stage reads from.

**Failure mode this stage exists to prevent:** A model that reports 98% validation accuracy in the report but fails in the field-validation exercise TIH-IoT explicitly requires, because the validation set was contaminated by near-duplicate leakage from training.

---

## 2. Preprocessing

**Purpose:** Normalize raw, heterogeneous images (different phones, lighting, resolutions — TIH's field-collected images will not be lab-standardized like PlantVillage) into a consistent tensor format models can consume, without destroying diagnostic signal.

**Stages:**

1. **Resolution standardization.** Resize to a fixed input resolution matched to backbone requirements (e.g., 224×224 for MobileNetV3/ResNet, 384×384 for ViT variants under evaluation) — using aspect-ratio-preserving resize + center/pad rather than naive stretch, since stretching distorts lesion shape/texture, a real diagnostic feature.
2. **Colour space normalization.** Normalize channel-wise using dataset-computed mean/std (not ImageNet defaults blindly reused) — particularly important for turmeric, where subtle leaf-blotch discoloration is a primary diagnostic signal, and colour-normalization choices materially affect what the model can distinguish.
3. **Region-of-interest extraction (leaf/vine segmentation).** Field images frequently contain background clutter — soil, other plants, hands, sky. A lightweight background-removal step (either classical: HSV-thresholding + GrabCut, or learned: a small pre-trained segmentation model) isolates the leaf/vine region before classification. This directly targets the "domain shift from lab to field" gap identified in the literature review — PlantVillage-trained models fail partly *because* they were never exposed to cluttered backgrounds during training, and partly because they never had to learn to ignore them.
4. **Quality gating.** Automatically flag and quarantine blurry (Laplacian-variance threshold), over/under-exposed (histogram check), or occluded images rather than silently training on them — a UG-appropriate, cheap check with outsized downstream benefit.
5. **Metadata-aware conditional handling.** Ridge gourd's dominant disease (ToLCNDV/Yellow Mosaic) presents as whole-leaf colour/pattern distortion, while turmeric's leaf blotch is localized lesion-based — so the ROI-extraction and augmentation *parameters* (not the pipeline structure) are tuned per crop, decided at this stage and logged per crop-config.

**Output:** A clean, standardized, quality-gated tensor dataset ready for augmentation, per crop.

---

## 3. Data Augmentation

**Purpose:** Synthetically expand effective training data diversity to (a) address class imbalance, and (b) close some of the lab-to-field domain gap *before* it's tested in field trials — augmentation choices here should simulate real deployment-time variation, not just inflate dataset size.

**Augmentation strategy, justified per transformation (not applied blindly):**

| Augmentation | Why it's included |
|---|---|
| Rotation (±25°), horizontal/vertical flip | Leaf orientation at capture time is arbitrary in field conditions — model must be rotation-invariant |
| Brightness/contrast jitter | Simulates variable field lighting (overcast vs. harsh midday sun) — directly targets the illumination-variance domain-shift gap |
| Random crop + zoom | Simulates variable camera-to-leaf distance by non-expert users capturing images in the field |
| Gaussian noise / slight blur | Simulates lower-end smartphone camera sensors likely to be used by smallholder farmers, not lab-grade cameras |
| CutMix / MixUp (class-conditional) | Regularizes decision boundaries between visually similar classes — important given ridge gourd YMD and other viral symptoms can be visually close to nutrient-deficiency discoloration |
| Class-balanced oversampling (not naive duplication — via augmentation-diversified copies) | Directly addresses the class-imbalance finding from the data-pipeline audit stage, rather than letting the model implicitly learn a majority-class prior |
| **Excluded: synthetic GAN-generated images**, at least for the initial model set | GAN-augmentation is flagged in recent literature as a promising future direction, not a validated production technique for disease-classification fidelity — using it as a *primary* augmentation source risks training on artifacts that don't reflect real disease morphology; this is deliberately scoped as a stretch goal for months 8–10, not baseline methodology |

**Critical constraint:** Augmentation is applied **only to the training split**, never to validation/test — a mistake surprisingly common in undergraduate ML work that silently invalidates reported metrics.

---

## 4. Feature Engineering *(applicable, scoped narrowly)*

For end-to-end CNN/ViT models, explicit feature engineering is largely superseded by learned representations — this stage is **not** hand-crafted-feature-based classification (that would be a regression to pre-2015 methodology and is not competitive). It is scoped to two justified additions:

1. **Colour-index features (optional auxiliary channel) for turmeric.** Since leaf-blotch and rhizome-disease correlation is a genuinely under-explored problem (identified as our novelty angle), we compute vegetation/disease colour indices (e.g., a Green-Red Vegetation Index variant) as an auxiliary input channel alongside RGB, to test whether it improves the model's ability to correlate subtle foliar colour shift with rhizome disease — this is evaluated empirically as one of the 3–5 model variants, not assumed to help.
2. **Symptom-pattern descriptors for ridge gourd ViT attention priors.** For the viral-mosaic detection problem, texture/pattern statistics (local binary patterns) are computed as a diagnostic tool during error analysis — used to *interpret* model failures (is the model missing mottling patterns specifically?), not necessarily fed into the model itself.

This stage is intentionally lightweight — the honest technical position for a 2026 CV project is that feature engineering's role is analysis and auxiliary-channel experimentation, not the primary modelling strategy.

---

## 5. Model Training

**Purpose:** Train the 3–5 candidate architectures per crop that the CFP explicitly requires, chosen to span the accuracy/efficiency/interpretability trade-off space identified in the literature review — not five arbitrary variants of the same architecture family.

**Candidate model set (per crop, justified individually):**

| Model | Role in comparison |
|---|---|
| **MobileNetV3-Small (transfer learning)** | Deployment-feasibility baseline — represents the edge-deployable end of the trade-off space |
| **EfficientNet-B0 (transfer learning)** | Mid-size accuracy/efficiency balance — the most commonly cited strong baseline in the literature |
| **ResNet-50 (transfer learning)** | Deeper CNN baseline — tests whether additional depth meaningfully improves fine-grained lesion discrimination over EfficientNet-B0 |
| **Vision Transformer (ViT-B/16 or a hybrid CNN-ViT)** | Tests the literature's claim that attention-based architectures generalize better to field-condition images than pure CNNs |
| **A lightweight custom/compressed variant (post hoc, from #5–6)** | Only finalized after comparison — this is the model actually optimized for deployment in stage 9, not trained from scratch redundantly |

**Training protocol:**
- Transfer learning from ImageNet-pretrained weights, with a two-phase schedule: (1) frozen backbone, train classifier head only, few epochs; (2) unfreeze top N blocks, fine-tune end-to-end at a lower learning rate. This is standard practice and avoids catastrophic forgetting of useful low-level filters when training data is modest in size (as TIH's per-crop labelled set is likely to be, relative to ImageNet scale).
- Class-weighted cross-entropy loss (weights derived from the class-distribution audit in Stage 1) to prevent majority-class collapse.
- Early stopping on validation macro-F1 (not accuracy — accuracy is a misleading stopping criterion under class imbalance).
- All experiments logged via MLflow or Weights & Biases: hyperparameters, metrics, model artifacts, git commit hash of training code — required for reproducibility and for the "technical reports documenting methodology" deliverable.

**Compute plan:** GPU access via TIH-IoT / IIT Bombay compute resources (to be confirmed with faculty mentor) or Colab Pro/Kaggle GPU tier as fallback for a 2-student UG team's realistic compute budget — flagged explicitly in the Resources section of the actual proposal, not assumed free.

---

## 6. Validation

**Purpose:** Measure whether a model actually works, at two distinct levels the CFP explicitly separates — computational validation and field validation. Conflating them is the single most common way agricultural CV projects overstate real-world readiness.

**6a. Computational validation:**
- **Stratified k-fold cross-validation** (k=5) on the TIH-provided dataset, respecting the group-aware split rule from Stage 1 (no leaf/plant-source leakage across folds).
- **Per-class metrics**, not aggregate accuracy: precision, recall, F1-score, and confusion matrix per disease class — required given the documented class-imbalance issue, where aggregate accuracy can mask an 8–15 percentage-point performance gap on minority classes.
- **Held-out field-condition test subset** — deliberately reserved from the noisier, less-curated portion of the TIH dataset (or created by intentionally including some low-quality-gated images that Stage 2 flagged) specifically to measure the lab-to-field generalization gap identified in the literature, rather than only reporting clean-condition performance.

**6b. Field validation (per CFP requirement):**
- Physical field visits with the PG fellow (agriculture background) and TIH-IoT domain experts to test the top-performing model(s) against live plants, comparing model predictions to expert diagnosis and, where available, Percent Disease Index (PDI) ground truth.
- Discrepancy logging: every model-vs-expert disagreement is recorded with the image and both judgments — this becomes both an error-analysis resource and, per the retraining loop, future training data.

**Output:** A validation report per model, per crop, with computational and field metrics reported separately and explicitly — never merged into a single misleading headline number.

---

## 7. Hyperparameter Optimization

**Purpose:** Systematically tune each of the shortlisted architectures rather than relying on default hyperparameters, which the literature shows can materially affect real accuracy (the earlier optimizer-comparison research found several percentage points of difference between Adam/RMSprop/AMSgrad on the same architecture and data).

**Method — justified choice, not default:**

- **Bayesian optimization (via Optuna)** over grid or random search, because the search space (learning rate, weight decay, augmentation intensity, batch size, unfreezing depth, label-smoothing factor) is large enough that grid search is computationally wasteful for a UG team's realistic compute budget, and random search wastes evaluations that Bayesian optimization would have skipped after early trials.
- **Search scope, per architecture:**
  - Learning rate (log-uniform range)
  - Weight decay
  - Fine-tuning depth (how many backbone layers unfrozen)
  - Batch size (constrained by available GPU memory)
  - Augmentation intensity (a meta-parameter over Stage 3's augmentation strength)
  - Label smoothing (relevant given class-imbalance and visually similar disease classes)
- **Optimization objective: validation macro-F1**, not accuracy — consistent with the validation-stage rationale.
- **Budget cap:** a fixed trial budget (e.g., 30–50 trials per architecture) is set upfront and documented, since unlimited tuning against a fixed validation set risks overfitting to that validation set itself — a subtlety worth being explicit about in a research proposal reviewed by faculty.

**Output:** One tuned configuration per architecture per crop, each logged with its full trial history for the technical report.

---

## 8. Model Comparison

**Purpose:** Produce the actual comparative-evaluation deliverable the CFP requires — a defensible, multi-axis leaderboard, not a single "winner" chosen on accuracy alone.

**Comparison axes (all reported together, per model, per crop):**

| Axis | Metric |
|---|---|
| Predictive performance | Macro-F1, per-class precision/recall, AUROC |
| Field robustness | Performance delta between clean-condition and field-condition test subsets (Stage 6a) |
| Computational cost | Inference latency (ms) on a representative mobile-class device/emulated hardware, model size (MB), FLOPs |
| Calibration | Expected Calibration Error — matters directly for the "confidence-threshold fallback to human expert" deployment design in Stage 9 |
| Interpretability quality | Qualitative + quantitative Grad-CAM localization sanity-check (does attention fall on the actual lesion, or on background artifacts?) — see Stage 9 |

**Selection rule, decided upfront (not post hoc):** The deployed model per crop is not simply the highest-F1 model; it's selected via a documented weighted scoring across all five axes, with weights justified in the proposal (e.g., field robustness and inference latency weighted highly given the CFP's explicit deployment mandate). Stating this rule *before* running the comparison, rather than picking whichever model happens to look best afterward, is what makes the comparison scientifically defensible rather than a foregone conclusion dressed up as an experiment.

**Output:** A leaderboard table per crop + written justification for the selected model, forming the core of the technical report deliverable.

---

## 9. Explainable AI

**Purpose:** Make the selected model's decisions interpretable — required both for the CFP's "validation through field trials and expert feedback" deliverable (experts need to see *why* a model made a call to trust or correct it) and for real farmer/extension-officer adoption, per the deployment-trust gap identified in the literature review.

**Techniques, chosen per model type:**

- **Grad-CAM / Grad-CAM++** for CNN-based models (MobileNet, EfficientNet, ResNet) — visualizes which image regions most influenced the classification, used both as an end-user-facing overlay and as an internal sanity check (does the model actually attend to the lesion, or is it exploiting a background shortcut — e.g., always predicting "diseased" when soil is visible because diseased-sample photos happened to be taken in muddier fields? This is a real, documented failure mode in field-image datasets, worth explicitly checking for).
- **Attention rollout / attention-map visualization** for ViT-based models — the transformer equivalent, since Grad-CAM's convolutional-gradient assumptions don't directly transfer.
- **Confidence calibration display**, not just a raw softmax score — presenting a calibrated confidence (from Stage 8's ECE measurement) alongside the prediction, with a defined low-confidence threshold that triggers a "refer to expert / TIH helpline" fallback message rather than a potentially wrong high-stakes recommendation (e.g., triggering unnecessary fungicide spend).
- **Per-class error gallery** compiled from field-validation discrepancies (Stage 6b) as a standing diagnostic tool for the team and for TIH-IoT reviewers, not a one-time report artifact.

**Output:** An explainability module integrated into the inference pipeline (not a separate offline analysis), producing a heatmap + calibrated confidence + optional fallback flag alongside every prediction.

---

## 10. Deployment

**Purpose:** Deliver the "deployment-ready inference pipeline for operational use" the CFP explicitly requires — designed for the real connectivity/compute constraints of Indian smallholder agriculture, not assumed cloud-always-on access.

**Architecture — hybrid, justified against the connectivity-constraint gap identified earlier:**

```
                     ┌─────────────────────────────┐
                     │      Mobile / Field Client    │
                     │  (Android app or lightweight  │
                     │   web capture interface)       │
                     └───────────────┬────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                        │
        [ONLINE — connectivity available]        [OFFLINE — no connectivity]
                 │                                        │
                 ▼                                        ▼
     ┌───────────────────────┐            ┌────────────────────────────┐
     │  Inference API          │            │  On-device quantized model  │
     │  (FastAPI + selected     │            │  (TFLite / ONNX Runtime      │
     │   model, containerized)  │            │  Mobile, INT8 quantized)     │
     └───────────┬─────────────┘            └──────────────┬─────────────┘
                 │                                          │
                 ▼                                          ▼
     ┌───────────────────────────────────────────────────────────────┐
     │        Prediction + Grad-CAM overlay + confidence score          │
     │        (queued locally if offline; synced when connectivity      │
     │         resumes, feeding Monitoring in Stage 11)                  │
     └───────────────────────────────────────────────────────────────┘
```

**Key engineering decisions, justified:**
- **On-device inference as the primary path, not a fallback** — given the explicit rural-connectivity constraint from the literature review, an architecture that *requires* cloud access for every prediction is not realistically deployable at the CFP's target user base. The selected lightweight model (from Stage 8's comparison, weighted toward inference latency and model size) is quantized (post-training INT8 quantization, validated for accuracy degradation before shipping) and converted to TFLite/ONNX Runtime Mobile for offline execution.
- **API path retained for higher-accuracy/heavier models** where connectivity permits (e.g., TIH field-office kiosks, extension-worker tablets with better connectivity) — served via a containerized FastAPI service, versioned per model release.
- **Prediction queuing and sync**, not silent failure — offline predictions are logged locally and synced when connectivity resumes, both to give the farmer their result and to feed the Monitoring stage's data-drift detection.
- **Versioned model registry** — every deployed model artifact is tagged with its training data version (Stage 1), architecture, and evaluation metrics, so a regression can always be traced to a specific change.

---

## 11. Monitoring

**Purpose:** Detect model degradation and data drift after deployment — the stage most agricultural CV projects skip entirely, and precisely the stage that operationalizes the CFP's "refinement based on newly available ground-truth data" deliverable.

**What's monitored:**

1. **Input data drift.** Track distributional statistics (image brightness/colour histograms, detected background clutter rate) of incoming field images against the training distribution — a growing divergence signals the model is seeing conditions it wasn't trained on (e.g., a new growing season, a new region with different lighting/soil colour), well before accuracy visibly drops.
2. **Prediction confidence drift.** Track the rolling distribution of model confidence scores over time — a rising rate of low-confidence predictions is an early warning signal, actionable before ground-truth labels are even available to compute accuracy directly.
3. **Expert-disagreement rate.** Where field-validation or extension-officer feedback is available (Stage 6b's ongoing discrepancy logging), track disagreement rate as a proxy for real-world accuracy over time — this is the most direct signal available but has the highest latency, since it depends on expert feedback being collected.
4. **System health metrics** (standard production ML-ops, not agriculture-specific): inference latency, API uptime, on-device crash/failure rate.

**Retraining trigger policy** (decided explicitly, not ad hoc): a documented threshold on drift/disagreement metrics triggers a retraining cycle — new field-validated data flows back into Stage 1's Data Pipeline as a new dataset version, closing the loop shown in the system diagram at the top of this document. This is what actually satisfies the CFP's continuous-improvement deliverable, rather than treating it as a one-line promise in the proposal text.

---

## 12. Documentation

**Purpose:** Deliver the CFP's explicit documentation requirements — technical reports, SOPs, and knowledge-transfer artifacts — as a first-class pipeline stage, not an afterthought written in the final week.

**Deliverables, mapped directly to CFP language:**

- **Technical reports** — methodology, per-model per-crop performance (from Stage 8's leaderboard), validation results (computational + field, Stage 6), and recommendations for operational implementation. Generated incrementally per milestone, not written once at the end.
- **Standard Operating Procedures (SOPs)** — covering: how to add a new disease class, how to trigger retraining, how to validate a new model version before deployment, how to interpret the confidence-threshold fallback — written so a future TIH-IoT team member (not the original fellows) can maintain the system, per the CFP's explicit "future model maintenance and scalability" requirement.
- **Dataset documentation** — a datasheet (following established datasheets-for-datasets practice) describing collection methodology, class distribution, known limitations, and licensing/IP status (noting TIH-IoT's retained ownership per the CFP's IP clause).
- **Reproducibility artifacts** — versioned code repository, experiment logs (Stage 5's MLflow/W&B tracking), and environment specification, so results are independently re-runnable, not just reported as numbers in a document.

---

## Summary: why this architecture, not a simpler one

A reviewer might reasonably ask why a UG fellowship needs this much infrastructure (versioning, drift monitoring, calibration, a retraining loop) rather than "train a CNN, report accuracy." The answer is that every non-obvious component above traces directly back to either (a) a documented research gap from the literature review — domain shift, class imbalance, poor field generalization, deployment compute constraints — or (b) an explicit CFP deliverable — comparative evaluation, field validation, deployment-ready pipelines, continuous improvement, SOPs. Nothing here is included for its own sake; the architecture is the direct engineering answer to the problem statement we've already established, and every stage should be traceable back to a specific sentence in the CFP's expected-outcomes list when we write the proposal.
