# Model Design & Comparison — Turmeric & Ridge Gourd Disease Detection
### TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This document specifies every candidate model family worth evaluating for this project, why each is included or excluded, and a scored comparison across the axes that matter for a 10-month UG fellowship with a deployment mandate. It closes with a recommended 3–5 model combination, justified against the CFP's requirements and the technical architecture already defined.

A note on method before the models: every "Expected Accuracy" figure below is a *literature-grounded estimate*, not a guarantee — drawn from the benchmark ranges established in our literature review (traditional ML: 80–90% on PlantVillage-style data; CNNs: 95–99%; transfer learning/transformers: >99% under controlled conditions, with a documented drop under field conditions). Actual figures on the TIH-provided turmeric/ridge gourd dataset will differ and must be measured empirically — stating this explicitly here is what keeps the eventual proposal honest rather than overpromising numbers no one has measured yet.

---

## Category A: Traditional Machine Learning

These are hand-crafted-feature + classical-classifier pipelines: feature extraction (colour histograms, GLCM texture features, HOG, LBP) followed by SVM, Random Forest, or Gradient Boosting.

### A1. SVM / Random Forest on hand-crafted features (colour + texture)

| Axis | Assessment |
|---|---|
| **Advantages** | Extremely low compute requirement (trains on CPU in minutes); fully interpretable feature importances; works with very small datasets, which matters if per-class sample counts for rarer turmeric/ridge gourd diseases are small; near-zero risk of overfitting to background artifacts the way deep nets can, since features are explicitly hand-specified |
| **Disadvantages** | Ceiling on achievable accuracy — cannot learn hierarchical/compositional features; brittle to lighting, angle, and background variation (exactly the field-condition variability we need to be robust to); requires manual feature-engineering effort and domain expertise to design good descriptors per crop |
| **Expected Accuracy** | 80–90% on clean/lab-condition images (literature-grounded); likely a larger field-condition accuracy drop than deep models, since hand-crafted features are more sensitive to the exact variability deep nets learn to ignore |
| **Training Time** | Minutes (CPU) |
| **Inference Speed** | Very fast (<10ms on CPU, no GPU needed) |
| **Deployment Difficulty** | Very low — trivially deployable on any device, including feature-phone-adjacent hardware |
| **Memory Usage** | Very low (<5MB model size typical) |
| **Research Value** | Low as a standalone contribution (well-established, unlikely to be novel), but **high value as a mandatory baseline** — the CFP asks for comparative evaluation, and a classical-ML baseline is the correct scientific control to demonstrate deep learning's actual marginal benefit rather than assuming it |

**Verdict:** Include as the mandatory baseline, not as a deployment candidate.

---

## Category B: Convolutional Neural Networks (trained from scratch or lightly pretrained)

### B1. Custom lightweight CNN (3–5 conv blocks, trained from scratch)

| Axis | Assessment |
|---|---|
| **Advantages** | Full architectural control — can be tailored to the specific input resolution/complexity needed for leaf-blotch (turmeric) vs. mosaic-pattern (ridge gourd) detection; smallest possible footprint since it's built for exactly this task with no wasted general-purpose capacity from ImageNet pretraining; excellent learning exercise for a UG team to understand CNN fundamentals |
| **Disadvantages** | Requires substantially more labelled data to train well from scratch than transfer learning, and TIH's per-crop dataset (even if generous) is unlikely to match ImageNet-scale data — from-scratch CNNs on comparatively small agricultural datasets have moderate historically-observed accuracy ceilings and materially higher overfitting risk; no benefit from features already learned on large-scale natural-image data |
| **Expected Accuracy** | 85–93% (literature-grounded for small-data from-scratch CNNs; noticeably below transfer-learned equivalents on the same data) |
| **Training Time** | Moderate (1–3 hours on a single GPU for a dataset of this scale, more epochs needed than transfer learning) |
| **Inference Speed** | Fast (depends on depth; comparable to MobileNet-class if kept shallow) |
| **Deployment Difficulty** | Low — small custom architectures convert cleanly to TFLite/ONNX |
| **Memory Usage** | Low (typically 2–15MB depending on depth) |
| **Research Value** | Moderate — demonstrates the data-efficiency gap versus transfer learning empirically (a legitimate comparative finding), but limited standalone novelty since the architecture pattern itself is well-established |

**Verdict:** Optional inclusion — most valuable as an empirical demonstration *within* the comparison (from-scratch vs. transfer-learned, same architecture family) rather than as one of the final deployed candidates.

---

## Category C: Transfer Learning (ImageNet-pretrained backbones, fine-tuned)

This is the most literature-validated category for this problem class, per our review.

### C1. MobileNetV3-Small / MobileNetV2 (transfer learning)

| Axis | Assessment |
|---|---|
| **Advantages** | Purpose-built for mobile/edge deployment (depthwise separable convolutions, squeeze-and-excite blocks); strong accuracy-per-FLOP ratio; extensive prior agricultural deployment precedent in the literature (e.g., guava leaf disease mobile deployment); quantizes cleanly to INT8 with minimal accuracy loss |
| **Disadvantages** | Accuracy ceiling below deeper architectures on fine-grained distinctions (e.g., distinguishing early-stage vs. mid-stage lesion severity); can underperform on subtle colour-pattern discrimination relevant to ridge gourd's viral mosaic symptoms, which benefit from more capacity |
| **Expected Accuracy** | 93–97% on clean conditions; moderate field-condition drop expected (smaller capacity generalizes less robustly to unseen variation than larger backbones, per the literature's domain-shift findings) |
| **Training Time** | Fast (30–60 min on a single GPU with frozen-then-finetune schedule, given ImageNet pretraining) |
| **Inference Speed** | Very fast (5–15ms on mobile-class hardware) |
| **Deployment Difficulty** | Low — this is the reference architecture for mobile CV deployment; excellent TFLite/ONNX Runtime Mobile support |
| **Memory Usage** | Low (4–16MB quantized) |
| **Research Value** | Moderate — not novel as an architecture, but essential as the deployment-feasibility anchor of the comparison; the actual research contribution is in *how well it holds up under field conditions relative to heavier models*, which is a genuine open question for these two under-studied crops |

**Verdict:** **Include** — this is very likely the final deployment candidate pending Stage 8 comparison results.

### C2. EfficientNet-B0 (transfer learning)

| Axis | Assessment |
|---|---|
| **Advantages** | Best-in-class accuracy-per-parameter via compound scaling (balances depth/width/resolution); the most frequently cited strong baseline across the plant-disease literature we reviewed, giving strong grounds for direct comparison to published results; good middle ground between MobileNet's efficiency and ResNet's raw capacity |
| **Disadvantages** | More sensitive to training hyperparameters than MobileNet/ResNet (compound scaling makes it less forgiving of a mis-tuned learning rate); slightly heavier than MobileNet for edge deployment, though still reasonable |
| **Expected Accuracy** | 96–98.5% on clean conditions (literature-grounded, this is the most commonly reported strong baseline range) |
| **Training Time** | Moderate (45–90 min on a single GPU) |
| **Inference Speed** | Fast (10–25ms on mobile-class hardware) |
| **Deployment Difficulty** | Low-moderate — well supported by TFLite/ONNX, slightly larger than MobileNet but still edge-viable |
| **Memory Usage** | Low-moderate (8–20MB quantized) |
| **Research Value** | High — directly comparable to the majority of published plant-disease benchmarks, making it the strongest anchor for claiming "our results are consistent with / improve on published literature" in the technical report |

**Verdict:** **Include** — the primary accuracy-focused CNN candidate, and the most citable against prior work.

### C3. ResNet-50 (transfer learning)

| Axis | Assessment |
|---|---|
| **Advantages** | Residual connections make it robust to train even when fine-tuning deeply; well-understood failure modes (extensive prior literature); strong raw feature-extraction capacity useful for fine-grained lesion discrimination |
| **Disadvantages** | Larger and slower than MobileNet/EfficientNet-B0 for a similar or only marginally better accuracy in most published comparisons; heavier deployment footprint makes it a less natural fit for the CFP's edge-deployment mandate; diminishing returns on accuracy relative to EfficientNet-B0 in the majority of recent comparative studies we reviewed |
| **Expected Accuracy** | 96–98% on clean conditions — comparable to, not clearly better than, EfficientNet-B0 in most published comparisons |
| **Training Time** | Moderate-slow (1–2 hours on a single GPU) |
| **Inference Speed** | Moderate (25–50ms on mobile-class hardware — noticeably slower than MobileNet/EfficientNet-B0) |
| **Deployment Difficulty** | Moderate — deployable via TFLite/ONNX but a heavier footprint makes it a harder sell for the primary offline mobile path defined in the architecture |
| **Memory Usage** | Moderate-high (25–50MB quantized) |
| **Research Value** | Moderate — valuable as a "does more depth help on our specific crops" comparison point, but redundant with EfficientNet-B0 if compute/time budget is tight, since the literature doesn't show a strong reason to expect it to clearly outperform |

**Verdict:** **Optional / lower priority** — include only if the team's time budget comfortably allows a 5th model; if forced to cut one candidate for time, this is the first to drop, since EfficientNet-B0 already covers the "deeper CNN" comparison point at lower cost.

---

## Category D: Vision Transformers

### D1. ViT-B/16 (pretrained, fine-tuned)

| Axis | Assessment |
|---|---|
| **Advantages** | Global self-attention captures long-range spatial relationships that convolutions miss — directly relevant to ridge gourd's whole-leaf mosaic/mottling pattern, which is a distributed symptom rather than a localized lesion, exactly the kind of symptom attention mechanisms are theoretically well-suited to; the literature we reviewed specifically found transformer-based models generalizing better than CNNs on field-captured images |
| **Disadvantages** | Data-hungry — ViTs are known to underperform CNNs when fine-tuning data is limited, since they lack CNNs' built-in spatial inductive bias; larger model size and higher compute cost for both training and inference; more fragile to hyperparameter choices than CNNs, requiring more careful tuning |
| **Expected Accuracy** | 95–98% on clean conditions (comparable-to-slightly-below best CNNs when fine-tuning data is limited, per literature); the key claimed advantage is a **smaller accuracy drop under field conditions**, not necessarily a higher clean-condition ceiling — this distinction should be explicitly tested, not assumed |
| **Training Time** | Slow (1.5–3 hours on a single GPU, more epochs typically needed to converge well) |
| **Inference Speed** | Slower (40–80ms on mobile-class hardware without further optimization) |
| **Deployment Difficulty** | Moderate-high — ViT mobile deployment tooling is less mature than CNN tooling (TFLite/ONNX support exists but is less battle-tested); quantization can be less accuracy-stable than for CNNs |
| **Memory Usage** | Moderate-high (30–90MB depending on variant, before quantization) |
| **Research Value** | **High** — this is the genuinely interesting empirical question for the project: does attention-based architecture's claimed field-robustness advantage actually hold on turmeric and ridge gourd specifically, two crops with almost no prior CV literature to check this against? This is close to a first-of-its-kind test for these crops |

**Verdict:** **Include** — this is the architecture most directly tied to the project's novelty claim (testing generalization robustness, not just clean-condition accuracy), and should not be cut for time even though it's the most resource-intensive candidate.

### D2. Data-efficient ViT variants (e.g., DeiT, or a smaller ViT-S/16)

| Axis | Assessment |
|---|---|
| **Advantages** | Specifically designed to address ViT's data-hunger problem via distillation or smaller patch/embedding configurations — a more realistic fit for TIH's dataset scale than full ViT-B/16 |
| **Disadvantages** | Still less mature deployment tooling than CNNs; adds another model to train/tune within a fixed time budget |
| **Expected Accuracy** | 94–97%, similar profile to ViT-B/16 but potentially more stable with less data |
| **Training Time** | Moderate (similar to EfficientNet-B0, faster than full ViT-B/16) |
| **Inference Speed** | Moderate (20–40ms) |
| **Deployment Difficulty** | Moderate |
| **Memory Usage** | Low-moderate (15–40MB) |
| **Research Value** | Moderate-high, but overlaps substantially with D1's research question |

**Verdict:** **Optional** — a reasonable substitute for ViT-B/16 if the team's compute/time budget is tight and a smaller-data-footprint transformer is preferred, but don't run both D1 and D2 alongside everything else — pick one transformer candidate, not two, to stay within the CFP's 3–5 model range without diluting depth of analysis on any single model.

---

## Category E: Hybrid Models

### E1. CNN feature extractor + Transformer/attention head (hybrid CNN-ViT)

| Axis | Assessment |
|---|---|
| **Advantages** | Combines CNN's strong low-level spatial inductive bias (good with limited data) with attention's ability to model global/distributed symptom patterns — directly targeted at exactly the trade-off both D1's disadvantage (data-hunger) and the ridge-gourd distributed-symptom problem raise; literature we reviewed specifically calls out hybrid CNN-Transformer models as a promising and *currently under-benchmarked* direction |
| **Disadvantages** | More architectural complexity to implement and tune correctly for a UG team within a 10-month timeline; less standardized/off-the-shelf than pure CNN or pure ViT (more implementation risk); harder to cleanly attribute performance gains to the hybrid design vs. just more parameters |
| **Expected Accuracy** | Potentially the highest of all candidates on field-condition data specifically (95–98.5%, with the smallest expected lab-to-field accuracy drop) — but this is the least literature-validated estimate of any model here, precisely because it's the least-studied architecture class for this problem |
| **Training Time** | Slow (2–4 hours, most implementation and tuning overhead of any candidate) |
| **Inference Speed** | Moderate-slow (30–70ms, depends heavily on the specific hybrid design chosen) |
| **Deployment Difficulty** | Moderate-high — custom architecture means custom conversion/quantization work, less "recipe" support than standard architectures |
| **Memory Usage** | Moderate (20–60MB depending on design) |
| **Research Value** | **Highest of all candidates** — this is the closest thing to an actual novel contribution in the model-architecture space itself, directly extending an explicitly-flagged literature gap ("hybrid CNN–Transformer models... currently under-benchmarked") to two crops that have never been tested with this approach |

**Verdict:** **Include as a stretch/optimization-phase model**, not a first-milestone one — given its implementation risk and time cost, this should be attempted after the four more standard candidates are working and validated, positioned as the "optimization" step the CFP's own project flow describes (train multiple models → compare → optimize the best-performing approach), rather than attempted from day one alongside everything else.

### E2. Multi-task learning head (disease classification + severity regression, shared backbone)

| Axis | Assessment |
|---|---|
| **Advantages** | A single shared backbone (e.g., EfficientNet-B0) with two output heads — disease classification AND a severity/PDI regression output — directly supports the CFP's explicit mention of Percent Disease Index (PDI) as a target field-validation metric, giving one model that produces both an actionable class label and a severity score, rather than two separate systems |
| **Disadvantages** | Requires PDI-labelled (not just class-labelled) training data, which may only be partially available depending on what TIH-IoT actually provides; multi-task loss balancing is a genuine tuning challenge |
| **Expected Accuracy** | Classification accuracy comparable to the shared backbone's single-task equivalent (e.g., ~96–98% if built on EfficientNet-B0), contingent on PDI label availability |
| **Training Time** | Similar to the base backbone, marginal overhead for the second head |
| **Inference Speed** | Near-identical to the single-task backbone (second head is cheap) |
| **Deployment Difficulty** | Similar to the base backbone |
| **Memory Usage** | Near-identical to the base backbone |
| **Research Value** | High **if PDI-labelled data is available from TIH-IoT** — directly operationalizes a specific, named CFP requirement (PDI values, mentioned explicitly for the PG-fellow field-experiment role) in a way none of the other candidates do |

**Verdict:** **Conditional include** — worth proposing, but flag explicitly in the proposal as contingent on confirming PDI-level (not just class-level) label availability with TIH-IoT before committing team time to it.

---

## Full comparison table (at a glance)

| Model | Exp. Accuracy | Train Time | Inference | Deploy Difficulty | Memory | Research Value |
|---|---|---|---|---|---|---|
| A1. SVM/RF (hand-crafted) | 80–90% | Minutes | <10ms | Very Low | <5MB | Low (baseline only) |
| B1. Custom CNN (scratch) | 85–93% | 1–3 hrs | Fast | Low | 2–15MB | Moderate |
| C1. MobileNetV3 (TL) | 93–97% | 30–60 min | 5–15ms | Low | 4–16MB | Moderate |
| C2. EfficientNet-B0 (TL) | 96–98.5% | 45–90 min | 10–25ms | Low-Mod | 8–20MB | High |
| C3. ResNet-50 (TL) | 96–98% | 1–2 hrs | 25–50ms | Moderate | 25–50MB | Moderate |
| D1. ViT-B/16 | 95–98%* | 1.5–3 hrs | 40–80ms | Mod-High | 30–90MB | High |
| D2. DeiT/ViT-S | 94–97% | ~1 hr | 20–40ms | Moderate | 15–40MB | Mod-High |
| E1. Hybrid CNN-ViT | 95–98.5%* | 2–4 hrs | 30–70ms | Mod-High | 20–60MB | **Highest** |
| E2. Multi-task (class+PDI) | ~96–98% | ~1 hr | ~similar to backbone | Low-Mod | ~similar to backbone | High (conditional) |

*\*Field-condition robustness, not just clean-condition accuracy, is the differentiating claim for D1/E1 — to be measured, not assumed.*

---

## Recommended combination: 4 models (within the CFP's 3–5 range)

Rather than maximizing the count, this selection is built to cover every axis the comparison needs to say something meaningful about, with no redundant entries:

1. **A1 — SVM/Random Forest baseline.** Not because it will be competitive, but because a comparative-evaluation deliverable without a classical-ML control cannot actually demonstrate deep learning's marginal value — it can only assert it. Minimal time cost, high scientific-rigor payoff.

2. **C1 — MobileNetV3 (transfer learning).** The deployment-feasibility anchor. Given the CFP's explicit "deployment-ready inference pipeline" requirement and the architecture's hybrid on-device/API design (defined in the technical architecture document), this is very likely the actual shipped model, and needs to be in the comparison on its own terms, not inferred from EfficientNet's numbers.

3. **C2 — EfficientNet-B0 (transfer learning).** The accuracy-focused, most-literature-comparable CNN candidate. This is what lets the technical report credibly say "our results are in line with / better than published benchmarks," which matters for review credibility.

4. **D1 — ViT-B/16 (transfer learning).** The model most directly tied to this project's actual research question — does attention-based architecture's field-robustness advantage hold for turmeric and ridge gourd specifically. This is the highest-research-value inclusion precisely because it's the least already-known answer.

**If team capacity/timeline allows a 5th model** (recommended only after the above four are trained, validated, and compared — not in parallel from day one): add **E1, the hybrid CNN-ViT**, explicitly framed as the "optimization" phase the CFP itself describes (develop multiple models → compare → optimize the best-performing one), rather than as a fifth independent baseline. This is where the project's strongest novelty claim lives, but it's appropriately sequenced as a later-milestone stretch goal given its implementation risk, not a first-milestone commitment that could derail the required baseline comparison if it runs into difficulty.

**What I'd explicitly deprioritize and why:** ResNet-50 (C3) and the from-scratch custom CNN (B1) are the two most defensible cuts if time runs short — C3 rarely beats EfficientNet-B0 by a meaningful margin in the literature at a higher deployment cost, and B1's main value (demonstrating the transfer-learning data-efficiency gap) is a secondary finding, not a primary deliverable. Cutting these keeps the team's effort concentrated on the four models that each answer a distinct question the CFP or the field itself is actually asking.

**On E2 (multi-task PDI head):** worth raising with TIH-IoT / your faculty mentor early — if PDI-level ground truth is confirmed available in the provided dataset, this is a strong, CFP-aligned addition to propose as a secondary output of the EfficientNet-B0 backbone at near-zero additional engineering cost, rather than a fifth full model.
