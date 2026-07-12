# Explainable AI — Turmeric & Ridge Gourd Disease Detection
### TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This document specifies the explainability layer that sits alongside every model in the comparison — not as a post-hoc add-on run once at the end, but as an integrated component of both the evaluation methodology (interpretability was already scored as one of four weighted comparison axes in the evaluation-methodology document) and the deployment architecture (the confidence-threshold fallback mechanism defined in the technical architecture document depends directly on the outputs specified here). This document explains why that integration matters, and specifies each technique precisely enough to implement.

---

## 0. Why explainability is not optional for this project — stated before the techniques

It's worth being explicit about this before describing any single technique, because it's the argument that justifies the engineering effort this section requires.

**The trust problem is structurally different in agriculture than in most CV applications.** A model prediction here doesn't just inform a passive decision — it can trigger a real economic action: a farmer spending on fungicide/pesticide, uprooting a plant, or conversely, *not* treating a genuinely diseased plant because the model said "healthy." Both directions carry real cost. The literature review identified deployment-trust as an explicit, named gap in the field — a model that gives no explanation or confidence calibration is unlikely to change farmer behavior at all, regardless of its benchmark accuracy, which means an uninterpretable model can be scientifically excellent and practically useless simultaneously.

**Three distinct audiences need explainability, each for a different reason**, and this shapes which techniques matter where:
1. **The farmer/end-user** needs a simple, visual justification ("here's the part of the leaf that looks diseased") to trust and act on a recommendation — this doesn't require technical XAI literacy, just an intuitive overlay.
2. **The PG fellow / TIH-IoT domain expert**, during field validation (Stage 6b of the architecture document), needs to see *why* the model made a call in order to judge whether that reasoning is agronomically sound or a shortcut/artifact — this is explainability as a validation instrument, not just a UX feature.
3. **The research team itself**, during error analysis (covered later in this document), needs explainability to diagnose *why* a model fails on specific cases — is it a genuine hard case, a label-noise artifact, or a model exploiting a spurious correlation? This is explainability as a debugging tool.

No single technique serves all three audiences well, which is why this document specifies a layered set rather than one method applied uniformly.

---

## 1. Grad-CAM (Gradient-weighted Class Activation Mapping)

**What it does:** Uses the gradients flowing into the final convolutional layer to produce a coarse heatmap highlighting which spatial regions of the input image most influenced the model's prediction for a given class.

**Why it's the primary technique for the CNN candidates (MobileNetV3, EfficientNet-B0, ResNet-50):** it requires no architectural modification, works directly on any standard CNN, and produces an intuitive visual overlay — exactly the format needed for the farmer-facing audience identified above. It's also the most literature-validated explainability technique for this specific problem domain (used directly in prior guava-disease and grape-disease CV work reviewed earlier).

**Grad-CAM++ variant, used specifically where relevant:** standard Grad-CAM can struggle when multiple instances of the same class-relevant feature appear in one image (e.g., multiple separate lesion spots on a single leaf) — Grad-CAM++'s weighted gradient combination handles multiple-occurrence localization more reliably, and is used as the default over vanilla Grad-CAM specifically because both turmeric leaf-blotch and ridge gourd mosaic symptoms can present as multiple discrete regions on a single leaf, not one contiguous lesion.

**Critical use beyond the pretty overlay — the internal sanity-check role:** Grad-CAM's most important function in this project isn't the farmer-facing visualization, it's catching **shortcut learning**. A documented, real failure mode in field-image agricultural datasets is a model learning to associate a background cue (soil colour, a particular capture setting) with a class, rather than the actual lesion — for example, if diseased-sample photos happened to be captured in muddier field conditions more often than healthy samples during TIH's data collection, a model could learn "muddy background → diseased" as a spurious shortcut that achieves high validation accuracy while being agronomically meaningless and guaranteed to fail unpredictably in deployment. Every model in the comparison has its Grad-CAM output spot-checked against a sample of correct AND incorrect predictions specifically to catch this — attention falling outside the plant/leaf region entirely is treated as a red flag requiring investigation, not just a curiosity.

---

## 2. Integrated Gradients

**What it does:** Attributes the prediction to individual input pixels by integrating gradients along a path from a baseline (typically a black or blurred image) to the actual input image — producing a finer-grained, pixel-level attribution map than Grad-CAM's coarser spatial heatmap.

**Why it's included as a complement to Grad-CAM, not a replacement:** Grad-CAM's resolution is limited by the spatial resolution of the final convolutional feature map (often quite coarse, e.g., 7×7 upsampled), which is good enough for "which region of the leaf" but not for "which specific pixels/edges within that region." Integrated Gradients gives pixel-level attribution, useful specifically for the research-team debugging audience — e.g., confirming that the model is responding to lesion *texture/boundary* specifically, not just the general presence of discoloration in that region, a genuinely finer distinction that matters for validating the model has learned the actual diagnostic feature rather than a coarser correlate.

**Why it's used selectively, not for every prediction in production:** Integrated Gradients is computationally more expensive than Grad-CAM (requires multiple forward/backward passes along the integration path, typically 20–50 steps) and its output is noisier/harder for a non-technical end-user to interpret visually (fine-grained pixel attribution maps are less intuitive than smooth region heatmaps). It is used during **model development, validation, and error analysis** (Section 6 below) — not as part of the real-time farmer-facing inference output, where Grad-CAM's simpler, faster, more intuitive overlay is the deployed choice.

**Axiomatic property worth noting for the technical report:** Integrated Gradients satisfies completeness (attributions sum to the difference between the model's output on the actual input and on the baseline) and sensitivity axioms that Grad-CAM doesn't guarantee — a legitimate, citable methodological reason to include it as a rigor check alongside Grad-CAM rather than relying on Grad-CAM's intuitive-but-less-formally-grounded heatmaps alone for the technical report's interpretability claims.

---

## 3. Attention Maps (for the ViT / hybrid CNN-ViT candidates)

**What it does:** Visualizes the self-attention weights within the transformer's layers — which image patches attend most strongly to which other patches, and, via attention rollout, which patches contribute most to the final classification token's decision.

**Why Grad-CAM doesn't directly transfer to ViT:** Grad-CAM's formulation assumes a convolutional feature map with clear spatial correspondence to the input image, which a pure transformer architecture doesn't have in the same form (ViT operates on a sequence of patch embeddings with global self-attention, not a spatially-organized convolutional feature stack) — using vanilla Grad-CAM on a ViT either fails outright or produces unreliable attributions, an implementation detail worth being explicit about, since naively applying CNN-designed XAI tools to transformer architectures is a genuine and common mistake.

**Method used:** Attention rollout (aggregating attention weights across all transformer layers, accounting for residual connections) to produce a single interpretable attention map per prediction, visually comparable in format to the Grad-CAM heatmaps used for the CNN candidates — this comparability is a deliberate design choice, since the evaluation-methodology document requires interpretability quality to be judged consistently across all models in the comparison, which is only possible if the visual outputs are in a roughly comparable format.

**Why this matters for the project's actual research question, not just as a technical requirement:** recall from the model-design document that the ViT candidate's core justification is the literature's claim that attention mechanisms generalize better to field-condition images specifically because they can model distributed, non-local symptom patterns (directly relevant to ridge gourd's whole-leaf mosaic symptoms, as opposed to turmeric's more localized lesions). Attention maps are the direct empirical evidence for or against that claim — if the ViT's attention genuinely spreads across the distributed mottling pattern on a ridge gourd leaf in a way Grad-CAM on the CNN candidates doesn't capture as coherently, that's a concrete, visualizable finding for the technical report, not just an accuracy-number difference. This is explicitly planned as a qualitative comparison exhibit in the final report, not an afterthought.

---

## 4. Feature Visualization

**What it does:** Distinct from the per-image attribution techniques above (Grad-CAM, Integrated Gradients, attention maps, all of which explain *one specific prediction*), feature visualization examines what individual learned filters/channels or intermediate representations respond to *in general*, across the dataset — e.g., via activation maximization (synthesizing an input that maximally activates a specific filter) or, more practically for this project's scope, via t-SNE/UMAP projection of the penultimate-layer embedding space across the full test set.

**Why the embedding-space visualization is the practically useful version for this project:** full activation-maximization feature visualization (synthesizing optimal input images per filter) is a more research-heavy technique with limited direct payoff for a diagnostic classification project on a UG timeline. The more directly useful application here is **projecting the test set's learned embeddings into 2D (via t-SNE or UMAP) and colour-coding by true class** — this produces a single, powerful diagnostic visualization: well-separated, tight clusters per disease class indicate the model has learned a genuinely discriminative representation; overlapping or diffuse clusters between two specific classes directly visualizes *which* disease pairs the model finds genuinely hard to distinguish, complementing the confusion matrix from the evaluation-methodology document with a geometric, rather than purely tabular, view of the same problem.

**Why this is included for every model in the comparison, not just the final selected one:** comparing the embedding-space separation across MobileNetV3, EfficientNet-B0, and ViT-B/16 for the same two confusable classes gives a genuinely comparative research finding — does one architecture's learned representation separate ridge gourd's viral-mosaic-vs-nutrient-deficiency confusion pair more cleanly than another's? — which is exactly the kind of qualitative-plus-quantitative comparative evidence a strong technical report includes beyond raw metric tables.

---

## 5. Confidence Scores

**What it is:** The model's own predicted probability for its output class — but, critically, **calibrated** confidence, not the raw softmax output used naively.

**Why raw softmax confidence is explicitly insufficient, stated clearly:** modern deep neural networks are well-documented to be systematically overconfident — a raw softmax score of 0.98 does not reliably mean the model is correct 98% of the time; it's a common, well-established finding that networks (especially larger, more expressive ones) produce overconfident probability estimates that don't match their true empirical accuracy. Deploying raw softmax confidence directly into the fallback-threshold system described in the architecture document would give farmers a false sense of certainty exactly in the cases where the model is most likely to be silently wrong.

**Calibration method used:** Temperature scaling — a simple, single-parameter post-hoc calibration technique that rescales the softmax logits using a temperature parameter fit on a held-out calibration set (a subset of the validation split, distinct from the test set, to avoid any test-set leakage into this calibration step). Chosen over more complex calibration methods (e.g., Platt scaling per class, isotonic regression) because temperature scaling is simple, doesn't distort the model's class ranking (so it doesn't interfere with the accuracy/F1 metrics already measured), and is well-validated in the broader deep-learning calibration literature as an effective, low-overhead default.

**Expected Calibration Error (ECE)** is the metric used to quantify calibration quality — already specified as one of the axes in the evaluation-methodology document's model comparison; this section specifies *how* that calibration is actually produced, not just measured.

**Deployment use, concretely:** the calibrated confidence score is what actually feeds the confidence-threshold fallback mechanism ("if confidence is below threshold X, flag for expert review / TIH helpline referral rather than presenting a potentially-wrong high-stakes recommendation") — this is the direct, load-bearing connection between this XAI section and the deployment architecture, not a decorative feature.

---

## 6. Error Analysis

**What it is:** A structured, systematic review of every incorrect prediction on the test set (and, ongoingly, on field-validation discrepancies per Stage 6b of the architecture document) — not a passive byproduct of computing metrics, but an active investigative process using the techniques above as tools.

**Structured error-analysis protocol:**
1. **Categorize every error** into one of: (a) genuine model failure on a clear-cut case, (b) a genuinely ambiguous/borderline case even for a human expert (cross-referenced against the inter-annotator agreement data from the dataset-preparation document's label-verification stage), (c) a probable label-noise case (the "error" may actually be a mislabelled ground truth, not a model failure), or (d) an out-of-distribution/edge-case input (e.g., an unusual capture angle or an co-occurring symptom combination not well-represented in training data).
2. **For every category-(a) genuine failure**, apply Grad-CAM/attention-map visualization to check for shortcut-learning red flags (Section 1) before concluding it's a "hard case" — a model failing because it learned a spurious background correlation is a different, more fixable problem than a model failing on genuinely subtle symptom differentiation.
3. **Aggregate errors by class-pair** (which specific true-class/predicted-class confusion pairs dominate) — cross-referenced directly against the confusion matrix and the embedding-space visualization (Section 4), giving three independent views (tabular, geometric, and case-by-case qualitative) of the same underlying confusion pattern, which is more convincing evidence in a technical report than any single view alone.
4. **Feed category-(c) findings back into the dataset-preparation pipeline** as candidate re-verification targets (closing a loop with the label-verification stage of the dataset document) — error analysis here is not just a reporting exercise, it's an active input to improving the dataset itself, directly relevant to the CFP's continuous-improvement deliverable.

---

## 7. False Positive Analysis

**Definition in this context:** Cases where the model predicted a disease class but the true label was healthy (or a different, incorrect disease class) — a "false alarm."

**Why analyzed separately from false negatives, not folded into general error analysis:** false positives and false negatives have different practical consequences (Section A4 of the evaluation-methodology document already established this asymmetry) and, empirically, often have different root causes — worth investigating as distinct categories rather than one undifferentiated "errors" bucket.

**Specific investigative questions applied to every false positive:**
- Does Grad-CAM/attention localization fall on an actual leaf feature, or on background/artifact? (Shortcut-learning check, per Section 1.)
- Is the "false" positive actually an early-stage true positive not yet reflected in the ground-truth label (i.e., the model detected genuine early symptoms before the labelling process captured them)? This is a real, non-hypothetical possibility in agricultural labelling and worth checking against the source image and, where feasible, follow-up information, rather than assumed to always be a pure model error.
- Is there a specific visual confound present (e.g., water droplets, natural leaf-vein pigmentation, insect damage unrelated to the disease being classified) that a human expert would also plausibly misjudge from a single static image? This distinguishes "the model needs to be better" from "this is a genuinely hard single-image classification problem, possibly needing multi-image or additional-context input in a future iteration" — an honest, useful distinction for the technical report's limitations section.

**Aggregate reporting:** false-positive rate broken down per class and per crop, cross-referenced with the confidence-score distribution of those specific false positives (are false positives concentrated near the decision threshold — a calibration/threshold-tuning issue — or confidently wrong — a more concerning representational issue) — this distinction directly informs whether the fix is threshold recalibration or additional training data/architecture changes.

---

## 8. False Negative Analysis

**Definition in this context:** Cases where the model predicted healthy (or an incorrect disease class) but the true label was a specific disease — a "missed detection."

**Why this receives the most scrutiny of any error category, stated explicitly:** per the recall-prioritization argument already made in the evaluation-methodology document, false negatives carry the highest real-world cost in this application — an undetected disease continues to spread and cause the yield loss quantified throughout the literature review, with no corrective action triggered at all. False negative analysis is therefore treated as the highest-priority error-analysis category, not evaluated with equal weight to false positives by default.

**Specific investigative questions applied to every false negative:**
- Is this an early-stage/subtle presentation where symptom expression is genuinely minimal — a fundamental image-based-diagnosis limitation (some diseases may simply not be visually detectable at very early stages, a real biological constraint no amount of model improvement can overcome, worth stating honestly rather than implying the system is infallible) — versus a clearer-symptom case the model should have caught?
- Does the confidence score for the (incorrect) predicted class sit close to the decision boundary with the correct class, suggesting the information needed was present but the model's discrimination was imprecise, versus a confidently-wrong prediction, suggesting a more fundamental representational gap?
- Is there a systematic pattern across false negatives — e.g., a specific disease class disproportionately missed, or a specific image-quality/capture-condition pattern (per the field-condition-vs-clean-condition tracking established in the dataset-preparation and training-strategy documents) — that points to a specific, addressable data or model gap rather than random noise?

**Direct deployment design consequence:** given false negatives' outsized cost, the confidence threshold for the "refer to expert" fallback (Section 5) is set asymmetrically — biased toward flagging borderline cases as "uncertain, please verify" rather than confidently declaring "healthy," specifically to avoid the higher-cost failure mode of a confident false negative reaching a farmer as reassurance to take no action. This is a concrete, evaluable design decision emerging directly from this error-analysis category, not a generic disclaimer.

---

## 9. Deployment Implications — pulling the whole XAI layer together

**What actually ships in the deployed inference pipeline, versus what stays a development/validation tool** — this distinction matters because not every technique in this document is meant for the same audience, as established in Section 0:

| Technique | Ships to farmer-facing app | Used in field-validation review (PG fellow / TIH experts) | Used only in development/error-analysis |
|---|---|---|---|
| Grad-CAM / Grad-CAM++ overlay | ✅ (simplified visual) | ✅ (full detail) | — |
| Attention maps (ViT/hybrid) | ✅ (if ViT/hybrid is deployed) | ✅ | — |
| Integrated Gradients | — | ✅ (validation tool) | ✅ |
| Feature visualization (embedding plots) | — | — | ✅ |
| Calibrated confidence score | ✅ (drives fallback trigger) | ✅ | ✅ |
| Error analysis / FP/FN review | — | ✅ (informs field-validation discussion) | ✅ |

**The concrete deployment artifact per prediction, shipped to the end-user interface:**
```
┌───────────────────────────────────────────┐
│  Prediction: [Disease Class Name]           │
│  Confidence: [Calibrated %] [High/Med/Low]  │
│  Visual: [Grad-CAM/attention overlay on the │
│           original leaf image]              │
│                                              │
│  IF confidence < threshold:                 │
│   ⚠ "This result is uncertain. Please       │
│      consult your local extension officer   │
│      or the TIH-IoT support line before     │
│      taking action."                        │
└───────────────────────────────────────────┘
```

**Why this matters for the deployment architecture beyond the obvious UX benefit:** this design directly operationalizes the "trust and adoption" gap identified in the literature review — a farmer shown a heatmap that visibly corresponds to the actual diseased-looking part of their plant has a concrete, self-verifiable reason to trust (or appropriately distrust) the recommendation, rather than being asked to accept an opaque black-box label. This is also what protects against the worst deployment failure mode: a confidently-wrong prediction with no interpretability layer, presented to a farmer as unqualified fact, leading to a costly wrong action with no way for the farmer to have caught the error themselves.

**Consequence for the SOP documentation deliverable (per the architecture document):** the error-analysis findings (Sections 6–8) and the shortcut-learning checks (Section 1) are documented as a standing part of the model-maintenance SOP — future retraining cycles (per the monitoring/retraining loop in the architecture document) are required to re-run this same explainability audit, not just check aggregate accuracy, before a new model version is approved for deployment. This closes the loop between explainability-as-a-one-time-report and explainability-as-an-ongoing-quality-gate, which is the more defensible framing for a system expected to keep improving over the fellowship's 10 months and beyond.
