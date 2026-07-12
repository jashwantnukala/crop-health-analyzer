# Dataset Preparation Pipeline — Turmeric & Ridge Gourd Disease Detection
### TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

The technical architecture document already placed dataset preparation inside the larger system (Stages 1–3). This document goes one level deeper: it's the operational specification for exactly what happens to a raw image between "TIH-IoT hands it to us" and "it enters a training batch," and why each step exists rather than being generic ML housekeeping. Every step below is justified against either a specific failure mode documented in the plant-disease CV literature, or a specific property of turmeric/ridge gourd field images.

A guiding principle stated upfront, because it shapes every decision below: **the dataset pipeline's job is not just to make images "clean" — it's to make the training distribution honestly resemble the deployment distribution.** Most of the lab-to-field generalization gap identified in our literature review is not a modelling failure at all — it's a dataset-preparation failure, where training data was over-cleaned relative to what the model will actually see in a farmer's hands. Several steps below exist specifically to *avoid* over-sanitizing the data.

---

## 0. Pipeline overview

```
Raw TIH-IoT Images + Labels
        │
        ▼
┌───────────────────┐
│ 1. Image           │  → format/size/corruption checks
│    Preprocessing   │
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ 2. Noise Removal   │  → denoise WITHOUT destroying lesion texture
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ 3. Data Cleaning   │  → dedup, mislabel detection, quarantine
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ 4. Label           │  → expert-verified, inter-annotator agreement
│    Verification    │
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ 5. Dataset         │  → group-aware, stratified, leakage-free
│    Splitting        │
└─────────┬──────────┘
          ▼
     ┌────┴─────┐
     ▼          ▼
┌─────────┐ ┌─────────────┐
│ 6a. Data │ │ 6b. Resizing │  → applied per-split, in this order
│ Balancing│ │ & Normalize  │
└────┬─────┘ └──────┬───────┘
     └───────┬───────┘
             ▼
┌───────────────────┐
│ 7. Data            │  → TRAIN SPLIT ONLY, never val/test
│    Augmentation     │
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ 8. Cross-Validation │  → group-aware k-fold, not naive k-fold
│    Setup            │
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ 9. Quality           │  → gate: nothing proceeds to training
│    Assurance Gate     │     without passing this checklist
└─────────┬──────────┘
          ▼
    Training-ready dataset
```

Note the ordering choice: **splitting happens before balancing and augmentation, not after.** This is one of the most common and most damaging mistakes in student ML projects — augmenting or oversampling before splitting lets synthetically-derived duplicates of the same source image land in both train and test, silently inflating validation accuracy. Every step from here on assumes splitting happened first.

---

## 1. Image Preprocessing

**What it covers:** format standardization, corruption/integrity checks, and basic structural validation — the "can this file even be used" layer, before any content-level decision is made.

**Steps and justification:**

- **Format unification.** TIH-IoT's field-collected images will likely arrive as a mix of JPEG (most phone cameras) and possibly PNG or HEIC (newer iPhones). All images are decoded and re-encoded to a single consistent format (JPEG, quality 95) early, because downstream libraries (OpenCV, PIL, TensorFlow/PyTorch data loaders) behave inconsistently across formats, and silent format-handling bugs are a disproportionately common source of "why did 40 images fail to load" debugging time in student projects.
- **Corruption/integrity check.** Every file is opened and fully decoded (not just header-checked) to catch truncated or corrupted images before they cause a training-loop crash hours into a run — cheap to check, expensive to discover mid-training.
- **EXIF orientation correction.** Phone cameras frequently store images with an EXIF rotation flag rather than physically rotating pixel data. If this is ignored, a portrait-captured leaf photo can be loaded sideways, and a model trained on inconsistently-oriented images either learns spurious orientation-dependent features or wastes capacity learning rotation invariance that should have been handled at ingestion. This is corrected once, here, before anything else touches the image.
- **Minimum resolution filtering.** Images below a defined resolution floor (e.g., <300px on the shorter side) are quarantined, not silently upscaled — upscaling fabricates pixel information that doesn't exist and can mislead both training and, worse, a reviewer looking at "clean-looking" but actually-interpolated training data.

**Why this stage exists as its own step, not folded into resizing:** structural validity is a binary gate (does this file work at all), while resizing (Step 6b) is a content-transformation decision. Conflating them means a corrupted file can silently become a black or noise image after a "successful" resize — worth catching separately.

---

## 2. Noise Removal

**What it covers:** reducing sensor noise and irrelevant high-frequency artifacts, *without* removing the texture information that is often the actual diagnostic signal.

**This is the step most tutorials get wrong for agricultural images specifically**, and it's worth explaining why: standard denoising (Gaussian blur, aggressive median filtering) is designed for general photography, where noise is unwanted and texture is often not the target signal. In plant disease detection, **texture is frequently the diagnostic signal itself** — lesion surface texture, powdery-mildew granularity, mosaic-pattern fine structure. Applying strong denoising can literally remove the thing the model needs to learn to detect.

**Approach used here:**

- **Mild edge-preserving denoising only** — bilateral filtering or non-local means at conservative parameters, which reduce sensor/compression noise while explicitly preserving edges and local texture structure, rather than Gaussian blur which smooths indiscriminately.
- **Denoising strength tied to source, not applied uniformly.** TIH-provided lab/curated images likely need minimal denoising (already clean); field-collected images (particularly any captured by the PG fellows during farm visits) may have more sensor noise from lower-end phone cameras or low-light capture and need slightly stronger — but still edge-preserving — treatment. This decision is logged per image-source, not applied as one global setting.
- **Explicit exclusion of aggressive artifact removal for compression blocking** unless clearly severe — mild JPEG blocking artifacts are themselves part of the realistic deployment distribution (farmers will submit compressed phone photos), so over-correcting them in training data creates another lab-to-field mismatch, mirroring the general principle stated at the top of this document.

**Why this step is necessary, stated plainly:** noisy or artifact-heavy images without any treatment can cause the model to latch onto sensor-specific noise patterns as spurious correlates of a class (a real, documented failure mode — e.g., if diseased-sample photos happened to be captured on a particular lower-quality device more often than healthy samples). But over-aggressive denoising removes real diagnostic texture and produces a training distribution that doesn't match real deployment images. The edge-preserving, source-aware approach here is the deliberate middle path.

---

## 3. Data Cleaning

**What it covers:** identifying and handling duplicate, near-duplicate, and structurally problematic images at the dataset level (as opposed to Step 1's per-file checks).

**Steps and justification:**

- **Exact duplicate removal** via the checksum step already established in the data pipeline (Stage 1 of the architecture document) — trivial but necessary, since exact duplicates across splits are a direct leakage vector.
- **Near-duplicate detection.** Perceptual hashing (pHash) or embedding-similarity clustering (e.g., using a pretrained backbone's feature space) to detect images that are visually near-identical — multiple photos of the same leaf from slightly different angles taken in the same capture session. These aren't necessarily removed (they add legitimate viewpoint diversity), but they are **grouped**, and the group-aware splitting in Step 5 ensures near-duplicates from the same session never split across train and test. This is one of the single most important steps in this entire pipeline, because near-duplicate leakage is the most common cause of a plant-disease paper reporting 99%+ accuracy that then fails badly on genuinely new images.
- **Background-only / non-leaf image detection.** A lightweight sanity classifier (or even manual spot-check for a dataset this scale) flags images that don't actually contain the target plant structure (e.g., a mislabelled soil/sky photo) — these are quarantined for manual review rather than silently trained on.
- **Outlier/anomaly flagging.** Images whose embedding falls far outside the per-class distribution in feature space are flagged for manual review — sometimes these are genuinely rare presentations worth keeping (valuable, not noise), and sometimes they're mislabelled or corrupted — the point of this step is surfacing them for a human decision, not auto-removing them, since auto-removing rare-but-real presentations would ironically make the model less robust to the field variability we're trying to prepare it for.

---

## 4. Label Verification

**What it covers:** confirming that the class label attached to each image is actually correct — arguably the highest-leverage quality step in the entire pipeline, since a model can only ever be as good as its labels, and this is exactly where the PG fellow's agricultural domain expertise (explicitly named in the CFP) becomes operationally essential.

**Steps and justification:**

- **Expert-in-the-loop verification for a stratified sample, not the whole dataset (which is impractical at scale).** A statistically meaningful random sample per class (e.g., a minimum sample size or percentage per class, weighted toward minority classes where a single mislabel has outsized effect) is reviewed by the PG fellow / TIH-IoT domain expert against the assigned label. This directly operationalizes the CFP's own language: *"utilize their agronomy and crop protection knowledge to accurately label disease and pest datasets."*
- **Confusable-class targeted review.** For disease pairs known to be visually similar (e.g., early-stage nutrient deficiency vs. early-stage viral mosaic in ridge gourd — a documented confusion risk since both present as leaf discoloration/mottling), a higher review rate is applied specifically to those classes, since mislabeling here is both more likely and more damaging to model performance than for visually distinctive classes.
- **Inter-annotator agreement measurement**, if more than one person is involved in original labelling or re-verification (e.g., Cohen's kappa between two reviewers on a shared sample) — quantifies label reliability itself, which is a legitimate and citable methodological result for the technical report, not just an internal QA step.
- **Label-noise-aware training as a fallback, not a substitute for verification.** Even after verification, some residual label noise is realistic to expect. Techniques such as label smoothing (already included in the hyperparameter search space from the architecture document) provide some robustness to residual noise — but this is explicitly a fallback, not a reason to skip careful verification, since no amount of noise-robust training compensates for systematically wrong labels on an entire subclass.

**Why this is necessary, stated plainly:** unlike most of the other steps, a label error doesn't just add noise — it can actively teach the model the wrong signal, and unlike an image-quality problem, a bad label survives every other cleaning step untouched.

---

## 5. Dataset Splitting

**What it covers:** partitioning verified, cleaned data into train/validation/test sets — designed specifically to prevent the leakage failure mode flagged repeatedly above.

**Method:**

- **Group-aware splitting.** Using the near-duplicate/session groupings established in Step 3, entire groups (not individual images) are assigned to a single split. No two images from the same capture session/source plant appear in different splits.
- **Stratified by class**, within the group constraint, to preserve class proportions across train/val/test — critical given known class imbalance.
- **Stratified by capture condition where metadata allows** (e.g., roughly proportional representation of TIH-curated vs. field-collected images in each split) — so the validation and test sets aren't accidentally easier or harder than what the model will see in practice.
- **Split ratio:** 70% train / 15% validation / 15% test, a standard, defensible ratio for a dataset of this scale — adjusted per crop if per-class sample counts for either turmeric or ridge gourd are small enough that 15% test would leave too few samples in rare classes for meaningful metrics (in which case k-fold cross-validation, Step 8, becomes the primary evaluation mechanism rather than a single held-out test set).
- **A separate, deliberately-preserved field-condition subset** (as established in the architecture document's validation stage) — some of the noisier, less-curated portion of the data is intentionally held out specifically to measure the lab-to-field gap, rather than folded uniformly into a single test set that might average that gap away.

---

## 6a. Data Balancing

**What it covers:** addressing the class-imbalance pattern that the literature review flagged as near-universal in agricultural datasets (documented 8–15 percentage-point F1 gaps between majority and minority classes on comparable datasets).

**Applied only within the training split**, after splitting (never before — balancing before splitting risks leaking synthetically-derived samples across splits).

**Techniques, applied as a decision tree rather than one default choice:**

1. **First choice — class-weighted loss.** Adjust the loss function's per-class weights inversely proportional to class frequency. This is the least invasive intervention (no data duplication or synthesis), and is the default starting point.
2. **If severe imbalance remains** (e.g., a minority class with very few samples even after weighting proves insufficient in validation) — **augmentation-diversified oversampling**: minority-class images are included multiple times per epoch, each instance receiving a different random augmentation (Step 7) so the model doesn't see literal duplicates, only diversified variants.
3. **Explicitly avoided as a default: naive duplication without augmentation** — this teaches the model to recognize specific images rather than the underlying class, and can worsen overfitting to minority classes rather than improving generalization.
4. **Considered but flagged as unvalidated for this domain: SMOTE-style synthetic interpolation in feature space.** Feature-space interpolation methods developed for tabular data don't have a well-validated equivalent for raw images at production quality — this is noted as a possible research extension (comparing against GAN-based synthesis, mentioned as a stretch goal in the architecture document) rather than baseline methodology.

---

## 6b. Resizing & Normalization

**What it covers:** the content-level standardization that makes images consumable by a fixed-input-size neural network, applied consistently but *computed from the training split only* to avoid statistical leakage.

**Resizing:**
- Aspect-ratio-preserving resize with padding (letterboxing) to the target input resolution (224×224 for CNN backbones, 384×384 for the ViT candidate), rather than naive stretch-to-fit — stretching distorts lesion shape and aspect ratio, which are themselves potentially diagnostic (e.g., elongated vs. circular lesion patterns).
- Resolution choice is tied to the specific backbone requirements established in the model design document — not a single blanket resolution applied regardless of which of the 3–5 models will consume it.

**Normalization:**
- Per-channel mean/std normalization, computed **from the training split's own statistics**, not blindly reused from ImageNet defaults — while ImageNet-pretrained backbones nominally expect ImageNet normalization statistics, computing and comparing dataset-specific statistics is a worthwhile empirical check (turmeric and ridge gourd leaf imagery has different characteristic colour distribution than ImageNet's general-purpose photo collection), and the choice (dataset-specific vs. ImageNet-default normalization) should be one of the ablation decisions logged and justified in the technical report rather than assumed.
- Computing statistics from the training split only, and applying the identical fixed statistics to validation/test, is essential — computing normalization statistics from the full dataset (including test data) is a subtle but real form of data leakage, since it lets test-set information influence how training data is scaled.

---

## 7. Data Augmentation

Covered in detail in the technical architecture document (Stage 3) — the operational rule that belongs specifically in this pipeline document is sequencing: **augmentation is applied on-the-fly during training data loading, to the training split only, after splitting and after balancing decisions are made** — never as a static pre-generation step that bakes a fixed set of augmented images into the dataset (which both wastes storage and reduces the effective diversity the model sees across epochs, since on-the-fly augmentation produces a new random variant every epoch rather than a fixed finite set).

---

## 8. Cross-Validation

**What it covers:** robust performance estimation, particularly important given the likely modest per-class sample sizes for two under-studied crops.

**Method:**
- **Group-aware stratified k-fold** (k=5, consistent with the validation methodology in the architecture document) — the same leakage-prevention logic from Step 5 applies here: folds are constructed respecting source-plant/session groupings, not naive random k-fold, or the same leakage problem re-appears one level down inside cross-validation itself.
- **Nested cross-validation consideration for hyperparameter tuning specifically** — if hyperparameter optimization (Optuna, per the architecture document) uses the same validation folds that final performance is reported on, the reported performance is optimistically biased (the model has been indirectly tuned to that specific validation data). Where compute budget allows, an outer/inner nested CV structure keeps a truly untouched outer fold for final reporting. Given a UG team's realistic compute constraints, a documented, explicitly-stated compromise (e.g., a single untouched final test set held out entirely from all tuning, with k-fold CV used only for model *selection*, not final reported numbers) is an acceptable and honestly-reported alternative — but it must be stated explicitly as a limitation in the technical report, not silently assumed away.
- **Cross-validation used for model comparison decisions (Step 8 in the architecture document's model-comparison stage), while the final reported numbers for the chosen model come from the untouched held-out test set** — keeping these two roles distinct is what prevents the comparison process itself from inflating the final headline metric.

---

## 9. Quality Assurance (final gate)

**What it covers:** a final, explicit checklist gate before any data is considered "training-ready" — QA treated as a distinct pipeline stage with a pass/fail outcome, not an implicit property that emerges from the earlier steps.

**Checklist, run and logged for every dataset version:**

- [ ] Zero exact duplicates remain (checksum-verified)
- [ ] Near-duplicate groups fully contained within a single split (no cross-split leakage)
- [ ] Class distribution per split matches target stratification within a defined tolerance
- [ ] Minimum per-class sample count in validation/test meets a defined threshold for meaningful metric computation (flagged if not, rather than silently reporting an unreliable metric on 3 samples)
- [ ] Label-verification sample size and inter-annotator agreement score documented and above an agreed threshold
- [ ] Normalization statistics computed from training split only (verified, not assumed)
- [ ] No augmentation applied outside the training split (spot-checked)
- [ ] Dataset version tagged, checksummed, and logged in the version-control system established in the architecture document
- [ ] A field-condition-representative subset explicitly identified and reserved for the lab-to-field gap measurement

**Why QA is a distinct, final, gated step rather than assumed:** every step above can individually be done correctly and still combine into a flawed dataset if, for example, an engineer re-runs one step in isolation later (e.g., re-generating a normalization statistic after adding new field data) without re-checking downstream consistency. A single, versioned QA gate that must pass before any dataset version is used for a reported training run is what makes the eventual technical report's numbers trustworthy and reproducible — directly serving the CFP's own emphasis on "quality assessment" and "support for dataset validation activities" as a named UG-fellow deliverable.

---

## Best practices summary — the principles behind the steps

1. **Split before you balance or augment.** Every leakage bug in this domain traces back to violating this ordering.
2. **Group-aware, not just class-stratified, splitting.** Random splitting alone is insufficient whenever multiple images can originate from the same physical source (plant, session, field visit).
3. **Don't over-clean relative to the deployment distribution.** Aggressive denoising, exclusively lab-quality curated training data, and removing all "noisy" field images all *look* like good practice but actively worsen real-world performance if they make training data cleaner than what the model will actually see in a farmer's hands.
4. **Compute statistics (normalization, class weights) from training data only.** Any statistic computed from data outside the training split is a leakage vector, however small it looks.
5. **Treat label verification as a first-class pipeline stage with a domain expert in the loop**, not an assumption that the provided labels are ground truth — this is precisely where the PG fellow's agronomy training is not just useful but structurally necessary to the pipeline's validity.
6. **Log everything as a versioned artifact, not a one-time script run.** Given the CFP's explicit continuous-improvement/retraining requirement, a dataset preparation pipeline that can't be re-run identically on a future data increment isn't actually satisfying that deliverable, however good the initial numbers look.
7. **Every cleaning decision should be justified against a specific failure mode, not applied by default.** This document has tried to name the specific failure each step prevents rather than listing generic "preprocessing best practices" — that specificity is what will make this section of the proposal read as researched rather than templated.
