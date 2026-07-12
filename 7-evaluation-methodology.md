# Evaluation Methodology — Turmeric & Ridge Gourd Disease Detection
### TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This document specifies exactly how every candidate model will be scored, and — separately, since this is the part most comparisons get wrong — how those scores will be combined into a fair, defensible comparison across models that differ substantially in size, architecture, and computational cost. Metrics are organized into two families: **predictive-quality metrics** (is the classification correct) and **operational metrics** (can this actually run where it needs to run) — treated as two distinct axes throughout, per the model-comparison framework already established in the technical architecture document.

---

## Part A — Predictive Quality Metrics

### A1. Confusion Matrix (the foundation everything else derives from)

**What it is:** An N×N matrix (N = number of disease classes per crop) where each cell [i,j] counts how many images of true class i were predicted as class j.

**Why it comes first, not last:** Every other predictive metric below (accuracy, precision, recall, specificity, sensitivity, F1) is a summary statistic *derived from* the confusion matrix. Reporting only the derived metrics without the underlying matrix hides exactly the information a reviewer needs to judge whether a model's errors are benign or dangerous — e.g., a model confusing two visually similar-but-both-treatable fungal diseases is a very different failure than a model confusing "healthy" with "diseased," which has direct consequences for whether a farmer takes unnecessary or, worse, no action.

**How it's used here specifically:**
- One confusion matrix per model, per crop, computed on the untouched test set (per the testing-strategy document's "evaluate once" discipline).
- Read row-wise (recall per class) and column-wise (precision per class) — both are reported, not just one, since they answer different questions (Section A3/A4 below).
- Specifically inspected for the two confusable-class pairs flagged in the dataset-preparation document (e.g., ridge gourd's viral mosaic vs. nutrient-deficiency discoloration) — a targeted qualitative review of exactly those cells, not just an aggregate scan of the whole matrix.

---

### A2. Accuracy

**Definition:** (Correct predictions) / (Total predictions).

**Why it matters:** It's the most intuitive, most commonly reported metric, and useful as a first-glance sanity check and for direct comparison against published literature benchmarks (which almost universally report accuracy, making it necessary for the "how do our results compare to prior work" section of the technical report).

**Why it is explicitly NOT used as the primary selection metric here:** established repeatedly across the earlier documents — under the class imbalance documented in the dataset-preparation pipeline, a model can achieve deceptively high accuracy by defaulting toward majority-class predictions while performing poorly on minority disease classes. Accuracy is reported for every model **for comparability with literature**, but the actual model-selection and early-stopping decisions (per the training-strategy document) are governed by macro-F1, not accuracy. Stating this distinction explicitly in the technical report is what prevents a reviewer from reasonably asking "why does your headline accuracy look good but the confusion matrix shows a struggling minority class."

---

### A3. Precision (per class and macro-averaged)

**Definition:** Of all images the model predicted as class X, what fraction actually were class X. TP / (TP + FP).

**Why it matters here specifically:** Precision answers "if the model says this leaf has disease X, how much can the farmer trust that specific call?" Low precision on a disease class means false alarms — a farmer might apply a fungicide or pesticide unnecessarily, a real economic cost the literature review flagged as a deployment-trust concern. This makes precision not just a standard ML metric but a directly actionable one for this application's cost structure.

**Reporting:** Per-class precision (essential, since different disease classes will have very different precision, especially minority classes) and macro-averaged precision (unweighted mean across classes, consistent with the macro-F1 philosophy already established — doesn't let majority classes dominate the summary number).

---

### A4. Recall / Sensitivity

**Definition:** Of all images that actually were class X, what fraction did the model correctly identify. TP / (TP + FN). (Recall and sensitivity are the same quantity; both terms are used depending on convention — recall in general ML usage, sensitivity in the clinical/diagnostic-testing tradition this application is conceptually closer to.)

**Why it matters here specifically, arguably more than precision for this application:** Recall answers "of all the actually-diseased plants, how many did the model catch?" A missed detection (false negative) means a real disease goes untreated until symptoms worsen or spread — directly costing the yield-loss outcomes quantified in the literature review (20–40% global crop loss from undetected/late-detected disease). For an early-warning agricultural tool, the cost asymmetry between a false positive (unnecessary caution) and a false negative (missed disease, allowed to spread) generally favors prioritizing recall, at least for the "healthy vs. diseased" first-pass triage decision — this is stated explicitly as a design consideration for the confidence-threshold/fallback system in the explainable-AI stage of the architecture document, not just an evaluation afterthought.

**Reporting:** Per-class and macro-averaged, same rationale as precision.

---

### A5. Specificity

**Definition:** Of all images that were actually NOT class X (i.e., truly negative for that class), what fraction did the model correctly identify as not-X. TN / (TN + FP).

**Why it's included as a distinct metric, not redundant with precision:** specificity and precision both relate to false positives but answer different questions — precision asks "when the model predicts X, how often is it right," while specificity asks "when the true condition is not-X, how reliably does the model avoid predicting X." In a multi-class setting (this project has multiple disease classes per crop, not simple binary healthy/diseased), specificity is computed per class in a one-vs-rest formulation, and is particularly informative for the highest-stakes class in each crop's taxonomy — e.g., for ridge gourd, specificity on the "healthy" class specifically tells you how often the model wrongly flags a healthy plant as diseased across ALL disease categories combined, which is a distinct and useful operational question from any single disease class's precision.

---

### A6. F1 Score

**Definition:** Harmonic mean of precision and recall: 2×(P×R)/(P+R).

**Why the harmonic mean specifically, not a simple average:** the harmonic mean penalizes imbalance between precision and recall more heavily than an arithmetic mean would — a model with precision 0.95 and recall 0.30 has an arithmetic mean of 0.625 (looks reasonable) but an F1 of ~0.45 (correctly reflects that a model missing 70% of actual disease cases is not a good model, regardless of how trustworthy its positive predictions are). This property is exactly why F1 was selected as the primary training/early-stopping/model-selection metric throughout the training-strategy document — it can't be gamed by optimizing one of precision/recall at the other's expense.

**Reporting:** Per-class F1, and **macro-F1 as the primary headline predictive-quality metric across this entire evaluation methodology** — consistent throughout all four prior documents, restated here as the anchor metric this whole evaluation section is built around.

---

### A7. ROC Curve

**What it is:** A plot of True Positive Rate (recall/sensitivity) against False Positive Rate (1 − specificity) across all possible classification confidence thresholds, per class (one-vs-rest for the multi-class setting).

**Why it matters here specifically:** Unlike accuracy/precision/recall/F1 (all computed at a single, fixed decision threshold — typically the argmax of the softmax output), the ROC curve shows model behavior across the **entire range of possible confidence thresholds**. This is directly relevant to this project's deployment design: the explainable-AI stage's confidence-threshold fallback ("if the model isn't confident, refer to a human expert") requires choosing a specific operating threshold, and the ROC curve is the tool that makes that choice principled rather than arbitrary — you can read off, for any desired false-positive-rate tolerance, what true-positive-rate the model achieves at that threshold, and pick the deployment threshold accordingly rather than defaulting to the untuned 0.5 cutoff.

**Reporting:** One ROC curve per class per model (one-vs-rest), with the operating point actually used in deployment marked explicitly on the curve — connecting the evaluation artifact directly to a real deployment decision, not leaving it as a disconnected chart.

---

### A8. AUC (Area Under the ROC Curve)

**What it is:** A single scalar summarizing the ROC curve — the probability that the model ranks a randomly chosen positive example higher than a randomly chosen negative example, across all thresholds.

**Why it matters as a complement to F1, not a replacement:** F1 is threshold-dependent (computed at one specific decision point); AUC is threshold-independent, summarizing the model's overall ranking/discriminative quality regardless of where the final deployment threshold ends up being set. This distinction matters concretely here: two models could have similar F1 at their respective default thresholds but different AUC — the higher-AUC model has more "headroom" to be re-tuned to a different operating point (e.g., if field validation later suggests the recall-precision tradeoff should shift) without a full retrain, which is a meaningfully different practical property than F1 alone reveals.

**Reporting:** Per-class AUC (one-vs-rest) and macro-averaged AUC, reported alongside macro-F1 in every model's summary card — used as a secondary confirmatory metric, not the primary selection criterion (macro-F1 retains that role, consistent with the training-strategy document), but flagged explicitly if AUC and F1 rankings diverge across models, since that divergence itself is informative and worth discussing in the technical report rather than silently picking whichever metric favors a preferred model.

---

## Part B — Operational (Deployment-Readiness) Metrics

These metrics answer a categorically different question than Part A: not "is the model accurate" but "can this model actually run where the CFP requires it to run." Per the deployment architecture already established, this is not a secondary concern — it's a first-class evaluation axis with equal weight in model selection.

### B1. Inference Speed / Latency

**Definition:** Wall-clock time for a single forward pass (image in, prediction out), measured separately for (a) the on-device quantized path and (b) the API-served path, per the hybrid deployment architecture.

**Why it matters here specifically, not just as generic best practice:** the architecture document's design explicitly relies on the deployed model completing inference fast enough to be usable in the field, on the specific hardware profile of the target users — a model with excellent accuracy but multi-second inference latency on a mid-range Android device fails the CFP's "deployment-ready inference pipeline" requirement regardless of its predictive metrics, and it's important this is measured, not assumed.

**Measurement protocol, for fairness across models:**
- Measured on **identical representative hardware** for every model — a fixed mid-range Android device profile (or an emulated equivalent with documented specs) for the on-device path, and a fixed server configuration for the API path — never comparing one model's laptop-GPU inference time against another's mobile-CPU inference time, which would make the comparison meaningless.
- Reported as **median and 95th-percentile latency** over a fixed number of repeated inference runs (e.g., 100 runs after a warmup period), not a single measurement — single measurements are noisy and unrepresentative of real usage variance.
- Measured **post-quantization** for the on-device candidates specifically, since quantized inference speed can differ substantially from the full-precision training-time speed, and it's the deployed artifact's latency that matters for this metric, not the training artifact's.

---

### B2. Model Size

**Definition:** Storage footprint of the serialized model artifact, measured in megabytes, both pre- and post-quantization.

**Why it matters here specifically:** Directly determines deployability on storage-constrained mobile devices and affects app download size / over-the-air update cost for the farmer-facing client — a concrete, non-abstract deployment constraint, not a theoretical one. Also indirectly correlates with RAM usage at inference time (Section B3), though not identically, so both are measured rather than one inferred from the other.

**Reporting:** Pre-quantization (FP32) and post-quantization (INT8) size reported side by side for every model, with the percentage size reduction from quantization noted — this is itself informative, since architectures respond differently to quantization (some CNN architectures quantize with minimal accuracy loss; ViT quantization is comparatively less mature and may show a larger accuracy-vs-size tradeoff, a distinction flagged in the model-design document that this metric will empirically confirm or refute).

---

### B3. Memory Consumption

**Definition:** Peak RAM usage during inference (distinct from model size on disk — includes intermediate activation tensors, not just stored weights).

**Why it's measured separately from model size:** a model's stored weight size and its peak runtime memory footprint are not the same quantity — architectures with large intermediate feature maps (common in some CNN designs, and in ViT's attention matrices which scale with sequence length) can have meaningfully higher peak memory usage than their stored model size alone would suggest, and peak memory (not average) is what determines whether the app crashes on a lower-RAM device — a real, binary deployment-readiness question, not a matter of degree.

**Measurement protocol:** Peak RAM profiled during actual on-device inference (not estimated from architecture parameter counts, which is an unreliable proxy for actual runtime memory behavior) on the same representative hardware profile established in Section B1.

---

### B4. Deployment Readiness (composite operational assessment)

**What it is:** Not a single measurable number like the metrics above, but a structured checklist-based assessment combining B1–B3 with qualitative deployment-engineering factors, producing a documented readiness verdict per model.

**Checklist components:**
- [ ] On-device latency below a defined usability threshold (e.g., under ~1 second for a farmer-facing interactive experience — the specific threshold to be validated with TIH-IoT/end-user expectations rather than assumed arbitrarily)
- [ ] Model size compatible with target app distribution constraints
- [ ] Peak memory usage compatible with target device RAM profile (with margin for the rest of the application, not just the model in isolation)
- [ ] Successful quantization with documented, acceptable accuracy degradation (a defined maximum acceptable drop, e.g., no more than 1–2 percentage points of macro-F1 loss from quantization — stated as a threshold decided in advance, not judged post hoc to favor a preferred model)
- [ ] Framework/tooling maturity for the target deployment path (TFLite/ONNX Runtime Mobile support quality — flagged as a real differentiator, since ViT deployment tooling is documented as less mature than CNN tooling in the model-design document, and this is where that difference becomes operationally concrete rather than theoretical)
- [ ] Offline-capable (per the hybrid architecture's connectivity-constraint requirement) — confirmed functional with no network dependency for the on-device path specifically

**Why this is a composite checklist rather than a single score:** deployment readiness genuinely is multi-dimensional, and collapsing it into one number would hide exactly the kind of tradeoff information (e.g., "this model passes every check except memory, which fails on the lowest-RAM target device") that should inform an actual engineering decision, not just a leaderboard ranking.

---

## Part C — Fair Cross-Model Comparison Methodology

This is the part of the evaluation methodology most agricultural CV papers handle poorly, and it's worth being explicit about, since a reviewer familiar with the field will specifically check for these failure modes.

### C1. The core fairness problem

Comparing a 4MB MobileNetV3 against a 90MB ViT-B/16 purely on accuracy is not a fair or complete comparison — of course a larger, higher-capacity model can often reach marginally higher accuracy; the question that actually matters for this project is whether that marginal accuracy gain is worth its operational cost, given the CFP's explicit deployment mandate. Every model comparison in this project is therefore reported and interpreted as a **multi-axis comparison, never collapsed prematurely into a single ranking number**, until a final, explicitly-justified selection decision is made.

### C2. Standardized evaluation conditions (equal footing across models)

- **Identical test set** for every model — the same untouched, group-aware-split test set (per the dataset-preparation document) is used for every candidate, no exceptions, no per-model custom splits.
- **Identical evaluation hardware** for operational metrics (Section B1–B3) — as stated above, no comparing one model's numbers from one device profile against another's from a different profile.
- **Identical preprocessing pipeline** at inference time for every model, with only the architecture-required input resolution differing (per the dataset-preparation document's per-backbone resolution note) — every other preprocessing decision held constant across models being compared.
- **Identical hyperparameter tuning budget** (same number of Optuna trials, same compute-time allocation per architecture, per the training-strategy document) — so no single model is unfairly advantaged by having received disproportionately more tuning effort than the others.
- **Same class-weighting/loss-function family** across the primary comparison (with optimizer/loss ablations run and reported as clearly separate side experiments, per the training-strategy document) — isolating architecture as the actual variable being compared, not conflating it with other simultaneously-changed factors.

### C3. Weighted composite scoring — the actual selection mechanism

Consistent with the model-comparison stage already defined in the architecture document, the final model-selection decision uses a documented, pre-registered weighted scoring formula combining:

| Axis | Metrics feeding in | Suggested weight rationale |
|---|---|---|
| Predictive quality | Macro-F1 (primary), AUC (confirmatory) | Weighted highest — a model that doesn't classify correctly is not useful regardless of speed |
| Field robustness | Performance delta between clean-condition and field-condition test subsets | Weighted highly given the CFP's explicit field-validation mandate and the literature's emphasis on this exact gap |
| Operational readiness | Latency, size, memory, deployment checklist pass rate | Weighted meaningfully, not as a tiebreaker only — given the CFP's explicit "deployment-ready" requirement, this is not secondary |
| Interpretability quality | Grad-CAM/attention-map localization sanity (from the explainable-AI stage) | Weighted lower but non-zero — a correct-but-uninterpretable model is a real deployment/trust liability, not a purely academic concern |

**Why weights are decided and documented BEFORE running the comparison, not after:** stated in the architecture document already, repeated here because it's the crux of fairness — if weights were chosen after seeing results, it would be trivially easy to reverse-engineer a weighting scheme that favors whichever model the team already preferred, which would make the entire comparative-evaluation exercise scientifically hollow. Pre-registering the weighting rationale (as this document does) is what makes the eventual model-selection defensible to a reviewing panel.

### C4. Reporting format — never a single leaderboard number in isolation

Every model's final reported result takes the form of a **model card**, not a single accuracy figure:

```
┌─────────────────────────────────────────────────────────────┐
│ MODEL CARD: [Model Name] — [Crop]                             │
├─────────────────────────────────────────────────────────────┤
│ PREDICTIVE QUALITY                                             │
│   Macro-F1: __   Accuracy: __   Macro-AUC: __                 │
│   Per-class P/R/F1: [table]     Confusion matrix: [attached]   │
│   Clean-condition vs. field-condition F1 delta: __              │
├─────────────────────────────────────────────────────────────┤
│ OPERATIONAL READINESS                                          │
│   Latency (median/p95, on-device): __ / __ ms                  │
│   Model size (FP32 / INT8): __ / __ MB                          │
│   Peak memory (on-device): __ MB                                │
│   Deployment checklist: __ / 6 passed                           │
├─────────────────────────────────────────────────────────────┤
│ INTERPRETABILITY                                                │
│   Grad-CAM/attention localization: [qualitative + spot-check]   │
├─────────────────────────────────────────────────────────────┤
│ COMPOSITE SCORE: __  (weights: predictive __%, field __%,      │
│                        operational __%, interpretability __%)   │
└─────────────────────────────────────────────────────────────┘
```

This format is used identically for every model in the comparison (SVM/RF baseline through the ViT and hybrid candidates), so the final technical report presents a genuinely comparable, side-by-side evidentiary basis for the model-selection decision — not a narrative that happens to favor one model with supporting numbers assembled after the fact.

---

## Summary — why every metric earns its place

| Metric | What it uniquely reveals | Would be missed if omitted |
|---|---|---|
| Confusion matrix | Exact error structure between specific class pairs | Which mistakes are benign vs. dangerous |
| Accuracy | Literature-comparable headline number | Ability to benchmark against published work |
| Precision | Trustworthiness of a positive prediction | False-alarm/unnecessary-treatment risk |
| Recall/Sensitivity | Coverage of actual disease cases | Missed-disease/undetected-spread risk |
| Specificity | Reliability of negative predictions | False reassurance risk on healthy plants |
| F1 | Balanced precision-recall summary, gameable-resistant | Hidden precision/recall imbalance |
| ROC curve | Behavior across ALL thresholds, not just default | Ability to principled-ly choose a deployment threshold |
| AUC | Threshold-independent ranking quality | Headroom to retune the deployment threshold later |
| Inference latency | Real-world usability on target hardware | Silent deployment failure despite good accuracy |
| Model size | Storage/distribution feasibility | App-store/OTA-update deployment blockers |
| Memory consumption | Peak runtime footprint, distinct from disk size | Crashes on lower-RAM target devices |
| Deployment readiness checklist | Multi-dimensional go/no-go engineering assessment | False confidence from a single passing metric |

No metric here is included as a checkbox exercise — each closes a specific gap the others leave open, and together they let a reviewer verify that "best model" in this project means best *for actual deployment against the CFP's stated goals*, not best on a single number chosen after the fact to favor a convenient answer.
