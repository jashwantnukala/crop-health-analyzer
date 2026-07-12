# AI Training Strategy — Turmeric & Ridge Gourd Disease Detection
### TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This document specifies exactly how each of the 3–5 models identified in the model-design comparison will actually be trained — every hyperparameter and procedural choice below is stated with its scientific justification, so the eventual technical report can defend each decision rather than presenting it as an arbitrary default. Where a choice differs by model (e.g., CNN vs. ViT), that's called out explicitly rather than assuming one strategy fits all architectures.

---

## 0. Training strategy at a glance

```
┌────────────────────────────────────────────────────────────────────────┐
│                      TWO-PHASE TRANSFER LEARNING                        │
│                                                                           │
│  Phase 1: Head-only training          Phase 2: Fine-tuning              │
│  ┌──────────────────────┐            ┌──────────────────────────┐      │
│  │ Backbone: FROZEN       │   ──────▶ │ Backbone: partially       │      │
│  │ (ImageNet weights)     │  unfreeze │ unfrozen (top N blocks)   │      │
│  │ Classifier head: TRAIN │            │ LR: reduced 10-50x         │      │
│  │ LR: higher (~1e-3)     │            │ Both head + backbone train │      │
│  │ Epochs: short (5-10)   │            │ Epochs: longer, early-stop │      │
│  └──────────────────────┘            └──────────────────────────┘      │
│                                                                           │
│  Governing loop across BOTH phases:                                     │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ Mixed-precision forward/backward → gradient clip → optimizer  │      │
│  │ step → LR scheduler step → validation macro-F1 check →         │      │
│  │ checkpoint if improved → early-stopping counter update         │      │
│  └──────────────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Training Schedule

**Decision:** Two-phase transfer learning schedule (frozen-backbone warmup, then progressive unfreezing) for every transfer-learning candidate (MobileNetV3, EfficientNet-B0, ViT-B/16); a single-phase schedule with a longer warmup for the from-scratch classical baseline is not applicable (SVM/RF has no epoch-based schedule at all — noted explicitly since it's structurally different from every other candidate).

**Scientific justification:**
- **Phase 1 (frozen backbone, head-only training)** exists because randomly-initialized classifier-head weights, if allowed to backpropagate through the full pretrained backbone from step one, produce large, noisy gradients that can catastrophically corrupt the useful low-level and mid-level filters ImageNet pretraining already learned (edge detectors, texture filters, colour-gradient responses) — features directly relevant to lesion/texture detection that we do not want to destroy before the head has learned anything sensible to backpropagate.
- **Phase 2 (progressive unfreezing, fine-tuning)** exists because the frozen backbone's ImageNet-derived features, while useful, were never trained to distinguish turmeric leaf-blotch texture from healthy-leaf texture, or ridge gourd mosaic patterns from natural leaf-vein patterns — some backbone adaptation is necessary to reach competitive accuracy. Unfreezing only the top N blocks (not the entire backbone) is a middle path: low-level filters (edges, colour gradients, basic textures) are generic enough to reuse unchanged; higher-level filters (which encode more dataset-specific compositional patterns) benefit from adaptation to our specific crops.
- **N (number of unfrozen blocks) is treated as a tunable hyperparameter**, not a fixed guess — swept during hyperparameter optimization (Section 12) rather than assumed, since the correct unfreezing depth depends on how different turmeric/ridge gourd leaf imagery is from ImageNet's natural-image distribution, which is an empirical question, not something to assert in advance.

**Schedule length:** Phase 1: 5–10 epochs (short, since only the head is training and it converges quickly on frozen features); Phase 2: up to 40–60 epochs, governed by early stopping (Section 9) rather than a fixed count — stated as a ceiling, not a target, since training to a fixed epoch count regardless of validation behavior is a common and avoidable source of wasted compute and overfitting.

---

## 2. Learning Rate Strategy

**Decision:** Discriminative learning rates across phases, combined with a cosine-annealing schedule with warm restarts within Phase 2, plus a short linear warmup at the very start of each phase.

**Scientific justification:**
- **Discriminative (phase-differentiated) rates:** Phase 1 uses a relatively higher learning rate (~1e-3) since only the randomly-initialized head is training and needs to move quickly from its random starting point. Phase 2 uses a substantially lower rate (~1e-5 to 1e-4, roughly 10–50× lower) for the unfrozen backbone layers specifically, because these already hold useful pretrained weights — large gradient steps here would overwrite that useful prior knowledge rather than gently adapting it (the well-documented "catastrophic forgetting" risk in fine-tuning literature). If a differential rate is supported by the framework, the head can retain a slightly higher rate than the newly-unfrozen backbone blocks even within Phase 2.
- **Linear warmup (first ~2–3% of steps in each phase):** starting from a very small learning rate and ramping up avoids the well-documented instability of large-batch/high-LR optimizers (particularly Adam-family optimizers, used here) at the very start of training, when gradient estimates are least reliable.
- **Cosine annealing (with optional warm restarts):** chosen over a step-decay schedule because cosine annealing's smooth, continuous decay has been empirically shown in the broader vision-transfer-learning literature to reach comparable or better final validation performance than step decay, without requiring the manual guesswork of choosing decay milestones — an important practical consideration for a UG team without extensive prior tuning experience to draw on for exactly when a step-decay schedule should drop.
- **ViT-specific note:** ViT-B/16 is more sensitive to learning-rate choice than CNNs (a documented property — transformers lack CNNs' spatial inductive bias, making them more reliant on well-tuned optimization to reach comparable performance), so the ViT candidate receives a wider learning-rate sweep in hyperparameter optimization (Section 12) rather than reusing the CNN-tuned range directly.

---

## 3. Optimizers

**Decision:** AdamW as the default optimizer for all deep learning candidates, with SGD+momentum evaluated as a documented comparison point during hyperparameter optimization, not assumed inferior without testing.

**Scientific justification:**
- **AdamW (Adam with decoupled weight decay)** is chosen over plain Adam because Adam's weight decay implementation is mathematically entangled with its adaptive learning rate in a way that makes the effective regularization strength depend on the gradient scale — AdamW decouples these, giving more predictable, better-behaved regularization, which matters given regularization is doing real work here (Section 8) against a comparatively modest dataset size.
- **Why not plain SGD by default:** SGD with momentum can reach excellent, sometimes marginally better generalization than Adam-family optimizers in some CNN literature, but requires substantially more careful learning-rate and momentum tuning to reach that performance reliably — a real cost given a UG team's realistic tuning-time budget within a 10-month timeline. AdamW's adaptive per-parameter learning rates make it more forgiving of a less-than-perfectly-tuned initial learning rate, which is the pragmatically correct default for this project's constraints — but this is explicitly **tested, not assumed**: the literature we reviewed found several-percentage-point differences between optimizers on plant-disease classification specifically (RMSprop, Adam, AMSgrad compared directly), so an optimizer ablation is included as one line item in the hyperparameter search rather than treated as a settled question.
- **Optimizer choice is fixed per architecture family for the main comparison** (to keep the model comparison in Section 8 of the architecture document scientifically clean — you can't attribute a performance difference to architecture if you also silently changed the optimizer), with the optimizer ablation run as a separate, explicitly-labelled side experiment rather than mixed into the primary leaderboard.

---

## 4. Loss Functions

**Decision:** Class-weighted categorical cross-entropy as the primary loss, with label smoothing (factor ~0.1) applied, and a documented note on why focal loss is a considered-but-not-default alternative.

**Scientific justification:**
- **Class-weighted cross-entropy** directly addresses the class-imbalance problem established in the dataset-preparation document — weighting each class's loss contribution inversely to its training-set frequency prevents the loss surface from being dominated by majority-class gradients, which is the primary mechanism by which imbalanced datasets produce models that silently underperform on minority classes despite reasonable-looking aggregate accuracy.
- **Label smoothing** softens hard one-hot targets (e.g., 0.9/0.1 split instead of 1.0/0.0) — justified specifically for this problem because several disease pairs are visually similar and genuinely ambiguous even to human experts at early symptom stages (a property noted directly in the label-verification discussion of the dataset document). Training a model toward absolute 100% confidence on inherently ambiguous cases encourages overconfident, poorly-calibrated predictions — directly counter to the calibration requirement established for the explainable-AI/confidence-threshold fallback design in the architecture document.
- **Focal loss — considered, not adopted as default.** Focal loss down-weights easy (already well-classified) examples and focuses gradient updates on hard/misclassified examples, which is theoretically attractive for exactly this kind of imbalanced, visually-confusable multi-class problem. It is not adopted as the default because it introduces an additional hyperparameter (the focusing parameter γ) that requires its own tuning, and class-weighted cross-entropy with label smoothing is a simpler, well-validated starting point. Focal loss is retained as a documented candidate for the hyperparameter search (Section 12) if class-weighted cross-entropy proves insufficient on minority classes during validation — an empirical decision, not a default assumption either way.
- **E2 (multi-task model, if PDI labels are confirmed available)** uses a compound loss: classification cross-entropy plus a regression loss (Huber loss, chosen over plain MSE for its reduced sensitivity to any labelling-noise outliers in PDI scores) for the severity head, combined with a tunable weighting term between the two — this is flagged explicitly since it's the one loss-function decision genuinely distinct from the rest of the model set.

---

## 5. Batch Size Selection

**Decision:** Largest batch size that fits in available GPU memory without requiring gradient accumulation, subject to a documented lower bound based on batch normalization stability, with gradient accumulation used as a fallback for the ViT candidate if memory is constrained.

**Scientific justification:**
- **Memory-bound, not arbitrarily chosen:** batch size here is primarily constrained by the realistic GPU resources available to a UG team (Colab Pro/Kaggle-tier GPUs or shared institutional compute, per the architecture document's resourcing note) — this is stated honestly rather than assuming access to large-batch-capable hardware the team may not actually have.
- **Lower bound justification:** batch normalization (used inside MobileNetV3, EfficientNet-B0, ResNet-50) computes running statistics per batch; very small batch sizes (e.g., below ~16) produce noisy batch statistics that can destabilize training — this sets a practical floor on batch size independent of what memory would technically allow.
- **ViT-specific consideration:** ViT-B/16 has a substantially larger memory footprint per sample than the CNN candidates at comparable resolution, making it the most likely candidate to require either a smaller batch size or gradient accumulation (simulating a larger effective batch size by accumulating gradients across several smaller forward/backward passes before an optimizer step) — this is planned for explicitly rather than discovered as a surprise mid-project.
- **Batch size is swept as part of hyperparameter optimization within a memory-feasible range**, rather than fixed to a single arbitrary value, since the interaction between batch size and learning rate (larger batches generally tolerate/require larger learning rates) is well-documented enough that treating them independently in tuning would be a methodological gap.

---

## 6. Epoch Selection

**Decision:** No fixed epoch count treated as the actual stopping criterion — a generous epoch ceiling (Section 1's 40–60 for Phase 2) paired with early stopping (Section 9) as the actual governing mechanism.

**Scientific justification:** Fixing a specific epoch count in advance (e.g., "we trained for 50 epochs") without reference to validation behavior is scientifically indefensible — the correct number of epochs is a function of when the model stops improving on held-out data, which cannot be known before training. Stating an epoch *ceiling* (to bound compute cost/time) while letting early stopping determine the *actual* stopping point is the methodologically correct framing, and is what the technical report should describe rather than a single fixed number.

---

## 7. Transfer Learning Strategy

Already substantially covered in Sections 1–2 (two-phase, discriminative-LR schedule) — the additional decision to document here is **source of pretrained weights and domain-gap awareness**:

- **ImageNet-1k pretrained weights** are used as the starting point for all CNN and ViT candidates — the standard, well-validated choice given no large-scale agricultural-image pretraining corpus is being used here (worth noting as a limitation: ImageNet is a general-object dataset, not agriculture-specific, so the domain gap between ImageNet's pretraining distribution and turmeric/ridge gourd leaf imagery is real, which is precisely why Phase 2 fine-tuning — not just frozen-feature extraction — is necessary rather than optional).
- **A documented consideration, not adopted by default:** initializing from a plant-disease-specific pretrained checkpoint (e.g., a model pretrained on PlantVillage or a similar public plant-disease corpus) instead of pure ImageNet weights, as a possible closer-domain starting point. This is flagged as a candidate ablation for the optimization phase (paired with the hybrid CNN-ViT stretch goal in the model-design document) rather than default methodology, since it introduces its own domain-gap risk (PlantVillage's lab-condition images have their own distribution mismatch with our field-oriented target, as extensively discussed in the literature review) that needs to be tested empirically rather than assumed beneficial.

---

## 8. Regularization

**Decision:** A layered regularization strategy — weight decay (via AdamW, Section 3), dropout in the classifier head, data augmentation (already the primary regularizer, per the dataset document), and label smoothing (Section 4) — combined rather than relying on any single technique.

**Scientific justification:**
- **Why layered, not single-technique:** with a comparatively modest per-crop dataset size (relative to ImageNet-scale pretraining data), overfitting risk is real and multi-faceted — different regularization techniques address different failure modes (weight decay constrains parameter magnitude generally; dropout specifically prevents co-adaptation of classifier-head units; augmentation directly expands effective data diversity; label smoothing addresses overconfidence specifically). Relying on just one leaves the others' specific failure modes unaddressed.
- **Dropout placement:** applied in the classifier head (typically 0.3–0.5 dropout rate, tuned), not throughout the pretrained backbone — backbone dropout during fine-tuning can interact poorly with the pretrained batch-normalization statistics and is a less standard, less validated practice than head-level dropout for transfer learning specifically.
- **Weight decay magnitude** is treated as a tunable hyperparameter (typically swept across 1e-5 to 1e-2 on a log scale) rather than a fixed default, since the correct strength depends on dataset size and model capacity, both of which differ across the 3–5 candidates being compared.
- **Explicitly avoided: excessive regularization that would suppress genuine signal** — this is stated because over-regularizing a model that's already data-constrained can push it toward underfitting, which is just as much a validation-metric problem as overfitting; the validation curve (train vs. validation loss divergence, or lack thereof) is the empirical check used to confirm regularization strength is appropriate, not assumed correct from the hyperparameter value alone.

---

## 9. Early Stopping

**Decision:** Early stopping on validation macro-F1 (not accuracy, not validation loss alone), with a patience window and a minimum-improvement threshold, restoring the best checkpoint rather than the final-epoch weights.

**Scientific justification:**
- **Why macro-F1, not accuracy, as the monitored metric:** established repeatedly across this document set — accuracy under class imbalance can plateau or even improve slightly while minority-class performance degrades, since majority-class predictions dominate the aggregate number. Macro-F1 (unweighted average of per-class F1) is sensitive to minority-class degradation in a way plain accuracy is not, making it the metric that actually reflects the model quality we care about for this problem.
- **Why not validation loss alone:** validation loss can continue to decrease slightly (the model becoming more confident on already-correct predictions) even after the metric we actually care about (classification quality across all classes) has plateaued — macro-F1 is a more direct proxy for the actual deliverable.
- **Patience window** (e.g., stop if no improvement for 8–10 consecutive epochs) balances against two failure modes: too short a patience risks stopping during normal training noise/plateaus before genuine further improvement; too long wastes compute and risks eventual overfitting before stopping triggers.
- **Restoring best-checkpoint weights, not final-epoch weights,** at the end of training — since training continuing past the point of peak validation performance (even before early stopping's patience window elapses) is common, and the deployed/reported model should be the empirically-best checkpoint, not whatever the weights happened to be when the loop terminated.

---

## 10. Mixed Precision Training

**Decision:** Automatic mixed precision (AMP) — FP16/BF16 compute with FP32 master weights — enabled by default for all deep learning candidates on GPU hardware that supports it.

**Scientific justification:**
- **Why this matters concretely for this project, not just as a generic best practice:** given the realistic, constrained compute budget (Colab/Kaggle-tier or shared institutional GPUs, as flagged in the architecture document), mixed precision's roughly 1.5–3× training speedup and reduced memory footprint on supporting hardware directly translates into being able to run more hyperparameter trials, more models, and the ViT candidate's larger memory footprint (Section 5) within the same wall-clock and compute budget — this is a resourcing decision with direct downstream effect on how much of the model-comparison and hyperparameter-optimization plan is actually achievable in 10 months, not a minor implementation detail.
- **FP32 master weights retained** (standard AMP practice) specifically to avoid the numerical precision loss that pure FP16 training can cause in gradient accumulation and weight updates, particularly relevant for the smaller/more sensitive gradients present in Phase 2's low-learning-rate fine-tuning (Section 2).
- **Loss scaling** (dynamic, standard in modern AMP implementations) is used to prevent small gradient values from underflowing to zero in FP16 representation — a standard, necessary companion technique to AMP, not optional.

---

## 11. GPU Utilization

**Decision:** Explicit compute-budget planning and utilization monitoring as a project-management discipline, not just a training-loop implementation detail.

**Scientific/practical justification:**
- **Data loading pipeline optimization** (parallel data loading workers, prefetching, on-the-fly augmentation running on CPU concurrently with GPU compute) is treated as a required part of the training setup, since a GPU-bound-but-data-loading-starved training loop wastes the compute budget this whole strategy is designed to use efficiently — an easy-to-overlook inefficiency in student ML projects that directly undermines the mixed-precision speedup gained in Section 10.
- **Utilization logging** (GPU memory and compute utilization tracked via the experiment-tracking tool established in the architecture document, e.g., Weights & Biases system metrics) — not for its own sake, but because a persistently low GPU utilization number is a diagnostic signal that the bottleneck is elsewhere (data loading, CPU-bound augmentation) and worth fixing before concluding a model or configuration is inherently slow to train.
- **Realistic budget allocation across the model set:** given the 5 candidate models (or 4 core + 1 stretch) and hyperparameter optimization trials per model (Section 12), a documented compute budget (approximate GPU-hours allocated per model, per phase) is planned upfront as part of the project timeline, rather than discovering partway through the fellowship that compute has run out before the full comparison is complete — this is a direct, practical risk-mitigation decision for a 10-month UG timeline.

---

## 12. Hyperparameter Tuning

Already substantially specified in the technical architecture document (Stage 7) — the additions specific to the training strategy itself:

- **Search space summary** (consolidating decisions from this document): learning rate (log-uniform, phase-specific ranges per Section 2), weight decay (log-uniform, per Section 8), unfreezing depth N (per Section 1), batch size (within memory-feasible range, per Section 5), label smoothing factor (per Section 4), dropout rate (per Section 8), and — as a side-experiment, not the primary sweep — optimizer choice (per Section 3) and loss function variant (cross-entropy vs. focal loss, per Section 4).
- **Method: Bayesian optimization (Optuna)**, objective = validation macro-F1, fixed trial budget per architecture (justified in the architecture document to avoid both wasted compute and overfitting to the validation set through excessive tuning).
- **Per-architecture search, not one shared search across all models** — since the correct hyperparameter ranges genuinely differ by architecture (e.g., ViT's learning-rate sensitivity noted in Section 2), a single shared search would either be too narrow for some models or too wide (wasting trials) for others.

---

## 13. Model Checkpointing

**Decision:** Checkpoint on every validation-metric improvement (macro-F1), retaining the best checkpoint plus a small rolling window of recent checkpoints (not every single epoch), with full checkpoint metadata logged.

**Scientific justification:**
- **Why not save every epoch:** storage cost scales with model size × epoch count × number of models × number of hyperparameter trials — for a project training multiple architectures across a tuning sweep, saving every epoch's weights is wasteful and unnecessary once early stopping (Section 9) already identifies the checkpoint that matters.
- **Metadata logged per checkpoint:** epoch number, validation macro-F1 (and per-class breakdown), training/validation loss, hyperparameter configuration, dataset version (per the versioning established in the architecture document), and git commit hash of the training code — this is what makes a checkpoint traceable and reproducible months later, directly serving both the technical-report and SOP documentation deliverables from the architecture document.
- **Checkpoint used for both final deployment AND ongoing field-validation comparison** (Stage 6b of the architecture document) — the checkpointing system is not just a training-time convenience, it's the artifact that field-validation results get attached to, so checkpoint identity/versioning discipline has downstream consequences for the whole project's evidentiary trail, not just training-loop bookkeeping.

---

## 14. Validation Strategy

Consolidating decisions already justified across this document and the dataset-preparation document, stated here as the unified protocol:

- **Primary validation metric: macro-F1**, with full per-class precision/recall/F1 and confusion matrix reported alongside (never macro-F1 alone, since it can still hide which specific classes are struggling).
- **Group-aware, stratified validation split** (per the dataset document) — no source-plant/session leakage between train and validation.
- **Validation performed at the end of every epoch** during Phase 2 (and periodically during Phase 1, though less critical given its short duration) — used both for early-stopping decisions (Section 9) and checkpoint selection (Section 13).
- **A held-out, deliberately field-condition-representative validation subset**, evaluated separately from the general validation set, to track the lab-to-field gap throughout training, not just at the very end — this allows catching a model that's overfitting to clean-condition images early, rather than only discovering the gap after full training and comparison is complete.

---

## 15. Testing Strategy

**Decision:** A strictly untouched final test set (never seen during training, validation-based early stopping, or hyperparameter tuning), evaluated exactly once per model as the final reported number, plus the field-validation exercise (Stage 6b of the architecture document) as an independent, real-world testing layer beyond the computational test set.

**Scientific justification:**
- **Why "evaluated exactly once":** repeatedly evaluating on a nominal "test set" and adjusting anything in response (architecture choice, hyperparameters, even manual judgment calls) converts that test set into a de facto validation set, silently reintroducing the same optimistic bias that a proper train/validation/test separation is meant to prevent. This is a subtle discipline failure that's easy to violate informally (e.g., "let's just peek at test performance to decide between these two final candidates") — explicitly prohibited here as a stated methodological rule, not left to individual judgment in the moment.
- **Two-tier testing (computational test set + field validation)** reflects the CFP's own explicit two-part validation requirement ("performance evaluation using standard assessment metrics AND validation through field trials and expert feedback") — treated here as genuinely distinct testing layers with different failure modes each catches: the computational test set catches statistical/modelling errors; field validation catches real-world deployment gaps (lighting, camera quality, non-expert capture technique) that no held-out split of the same dataset can reveal, however carefully constructed.
- **Reporting discipline:** final technical report presents computational test metrics and field-validation results side by side, explicitly, rather than blending them into a single number — consistent with the validation-stage principle already established in the architecture document, restated here because it governs the *final* reported numbers specifically, which is what reviewers will scrutinize most closely.

---

## Summary table — every decision at a glance

| Component | Decision | Primary justification |
|---|---|---|
| Schedule | Two-phase (frozen → progressive unfreeze) | Prevents destroying pretrained features before head converges |
| Learning rate | Discriminative + cosine annealing + warmup | Matches update magnitude to what each phase/layer needs |
| Optimizer | AdamW (default), SGD tested as ablation | Forgiving of imperfect tuning; decoupled weight decay |
| Loss | Class-weighted CE + label smoothing | Addresses imbalance + inherent class ambiguity |
| Batch size | Max-feasible within memory, floor for BN stability | Compute-constrained, but bounded by BN statistics |
| Epochs | Ceiling only; early stopping governs actual count | Fixed epoch counts are scientifically indefensible |
| Transfer learning | ImageNet init, two-phase fine-tune | No agriculture-scale pretraining corpus available |
| Regularization | Layered (weight decay + dropout + augmentation + smoothing) | Different techniques address different overfitting modes |
| Early stopping | Macro-F1, patience window, best-checkpoint restore | Accuracy/loss alone hide minority-class degradation |
| Mixed precision | AMP (FP16/BF16 + FP32 master weights) | Directly expands what's achievable in the compute budget |
| GPU utilization | Explicit budget planning + utilization monitoring | Prevents silent compute waste undermining the plan |
| Hyperparameter tuning | Per-architecture Bayesian optimization (Optuna) | Search space genuinely differs across architectures |
| Checkpointing | Best + rolling window, full metadata logged | Reproducibility and field-validation traceability |
| Validation | Macro-F1 + per-class metrics + field-condition subset | Surfaces the specific failure modes that matter here |
| Testing | Untouched test set, evaluated once + field trials | Prevents test-set leakage into tuning decisions |

Every row in this table is designed to be defensible under direct questioning from a reviewing professor — "why did you choose X" has an answer grounded in either a documented property of this specific problem (class imbalance, lab-to-field gap, ViT's data/LR sensitivity) or a general, citable principle of sound ML methodology, not a framework default adopted without examination.
