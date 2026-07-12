#Literature Review — Image-Based Crop Disease Detection
### Crops: Turmeric (*Curcuma longa*) & Ridge Gourd (*Luffa acutangula*)
### TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This document grounds the proposal in the existing literature on crop disease detection, structured around nine questions: why the problem matters, the scale of the challenge globally and in India, its economic impact, the weaknesses of manual diagnosis, what AI-based solutions already exist, where the current research gaps are, why computer vision is the right modality, the challenges of real-world deployment, and where the field is heading. Citations are grounded throughout, and points that directly shape later proposal decisions — crop choice, model-comparison design, and the deployment section — are flagged as they arise.

---

## 1. Why crop disease detection matters

Plant health is not a peripheral agronomic concern — it sits directly on the food-security critical path. Humans derive roughly 80% of caloric intake from plants, and up to 40% of global crop production is lost to plant pests and diseases every year, a threat serious enough that FAO frames it as a risk to the 2030 Sustainable Development Goals. Independent of the exact percentage (estimates vary by methodology, discussed below), the direction is consistent: disease is one of the largest controllable levers on global food output, and detection is the first link in that control chain — you cannot manage what you cannot diagnose in time.

## 2. Global and Indian challenges

Global figures cluster in overlapping but not identical ranges depending on methodology:
- FAO's headline figure: 20–40% of global crop production lost annually to plant diseases and pests.
- A narrower, disease-only estimate (not counting weeds/pests) puts losses closer to 10–16% of global harvest, while a 2019 five-crop study covering roughly half of global caloric intake found wheat losses of 10–28%, rice 25–41%, maize 20–41%, potato 8–21%, and soybean 11–32% from 137 pathogens and pests combined — useful because it shows loss rates are crop-specific, not a flat global number, which matters when justifying per-crop modelling rather than one universal model.

For India specifically, several independent sources converge on a similar band:
- Crop yield losses from insect pests, diseases, nematodes, weeds and rodents range from 15–25% in India, amounting to roughly ₹0.9–1.4 lakh crore per year (about USD 12–18.5 billion).
- A historical trend analysis found overall pest-and-disease losses in India rose from 7.2% in the early 1960s to 23.3% by the early 2000s, with disease alone contributing about 6.5 percentage points, against 9.5 for insects and 12.5 for weeds — disease is a real but not the largest single component, worth stating honestly in the background section rather than inflating disease-specific loss claims.
- Crop-specific Indian data exists too — e.g., tea plantations alone report an estimated 147 million kg annual crop loss and ₹2,865 crore in revenue loss from pest infestation — illustrating that national aggregates hide large crop-to-crop variance, again reinforcing why building separate models per crop, rather than one universal classifier, is the scientifically correct framing.

Climate change is a compounding, forward-looking challenge: warming and erratic rainfall are expected to expand pest/pathogen range and frequency, meaning models trained today need to remain maintainable and retrainable as disease pressure patterns shift — a point that justifies treating continuous improvement from new ground-truth data as a core deliverable rather than boilerplate.

## 3. Economic impact

At global scale, plant pest and disease losses cost the global economy over USD 220 billion annually, with invasive insects adding at least USD 70 billion more. In India, agriculture contributes close to 28% of GDP, so a 15–25% biotic-stress loss rate translates into a macroeconomically material drag, not just a farm-level nuisance — the correct framing for a funding justification, since it ties a computer-vision research project to national economic outcomes rather than only farmer convenience.

## 4. Problems with manual disease diagnosis

The default diagnostic pathway in most of India (and most smallholder systems globally) is visual inspection, either by the farmer or, when available, an extension officer or plant pathologist. This has three structural weaknesses worth stating precisely rather than gesturing at "it's slow":

- **Expertise scarcity and skewed distribution.** Plant pathologists are typically trained on a limited set of diseases per host and are concentrated near research institutions, not in the field. Multiple diagnosticians are often required for reliable multi-disease identification, and they are not always available in a timely manner, leading to costly delays.
- **Subjectivity and inter-observer variance.** Visual symptom assessment is inherently subjective, especially for early-stage or visually overlapping symptoms — the same lesion pattern can be conflated across different pathogens by non-specialists, or even across specialists without lab confirmation.
- **Reactive rather than preventive.** Manual scouting typically happens on a schedule (e.g., weekly extension visits) rather than continuously, so by the time a disease is confirmed, the epidemic window for cost-effective intervention may already be closing — early detection is repeatedly identified in the literature as the single highest-leverage point for loss minimization.

This is the direct motivation for automation: the goal isn't to replace expert judgment but to make first-pass triage available at the point and moment of need, independent of expert availability.

## 5. Existing AI solutions — what's already been tried

The dominant public benchmark is **PlantVillage** (Hughes & Salathé, 2016): 54,306 lab-captured images across 14 crops and 38 disease classes. On this benchmark, model performance has followed a fairly predictable maturity curve: traditional ML (SVM, RF, etc.) typically reaches 80–90% accuracy; CNN-based deep learning reaches 95–99%; and transfer learning / transformer-based approaches consistently exceed 99% accuracy under those controlled conditions. Recent comparative work has specifically benchmarked architectures like EfficientNet-B0, ResNet-50, MobileNetV2, and Vision Transformer (ViT) against each other — directly relevant precedent for a comparison spanning 3–5 alternative models per crop; it suggests the comparison should span at minimum a lightweight CNN (MobileNet-class, deployment-friendly), a deeper CNN (ResNet/EfficientNet-class, accuracy ceiling), and an attention-based architecture (ViT or hybrid CNN-ViT), rather than five arbitrary variants of the same family.

More recent work has also explored **zero-shot / vision-language approaches** — CLIP-style models that classify via textual disease descriptions without task-specific training — with transformer and CLIP-based zero-shot models showing superior generalization on field images captured by farmers under natural conditions compared to CNNs, though at typically lower peak accuracy than fully supervised models on in-distribution data. This is a genuine trade-off worth surfacing in the model-comparison rationale: accuracy-under-lab-conditions vs. robustness-under-field-conditions are not the same axis, and a proposal that only optimizes the former will not survive a field-validation requirement.

## 6. Current research gaps

The literature is fairly consistent about where the field is stuck, and these map cleanly onto what a reviewer will expect a proposal to address rather than ignore:

- **Domain shift from lab to field.** Most agricultural datasets like PlantVillage consist of standardized lab images that don't reflect actual field conditions, and models trained on such data show measurable performance drops when transitioned to field deployment. This is the single most cited gap and is exactly what a field-trial validation requirement is designed to force a proposal to confront.
- **Class imbalance and rare-disease underrepresentation.** In PlantVillage, one crop can account for ~27% of the dataset while minority classes fall below 1%, producing an 8–15 percentage-point F1 gap between majority and minority classes — directly relevant since any dataset spanning many crops will almost certainly have similar imbalance, and the model-comparison methodology needs to report per-class metrics, not just aggregate accuracy.
- **Poor cross-crop generalization.** Models tuned for one crop/disease taxonomy often don't transfer cleanly to another, which is one of the stated reasons crop-specific model development is mandated over a single universal classifier.
- **Edge/mobile deployment constraints.** Critical gaps identified include insufficient annotation standards, limited cross-domain validation, and computational constraints for edge deployment — i.e., the highest-accuracy architecture (large ViT/CNN ensembles) is frequently not the deployable one, and the literature increasingly treats this as a first-class design constraint rather than a post-hoc optimization step.
- **Limited interpretability.** Explainable AI (Grad-CAM, attention visualization) is called out as necessary but underused — important for building farmer/extension-officer trust in a system whose recommendations affect real input costs (pesticide, fungicide spend).

## 7. Why computer vision is the right modality

Three reasons converge, and it's worth being explicit that this isn't the only option (soil sensors, spectral/hyperspectral imaging, and molecular diagnostics all exist) — CV on RGB leaf imagery is the right choice for this specific program because:

- **Symptom expression is primarily visual.** Most fungal, bacterial, and many viral diseases manifest as visible leaf/fruit lesions, discoloration, or necrosis before yield impact is severe — so RGB image classification captures the actual diagnostic signal pathologists themselves use.
- **Acquisition cost and scalability.** A smartphone camera is already in nearly every Indian farming household or accessible via the nearest extension worker, unlike hyperspectral or molecular diagnostic equipment, which is expensive, lab-bound, and turnaround-slow. This is what makes CV-based diagnosis operationally deployable at smallholder scale, versus scientifically interesting but practically inaccessible alternatives.
- **Maturity of the toolchain.** Transfer learning from ImageNet-pretrained backbones (CNNs and ViTs) is well-understood, computationally tractable even with moderate-sized labelled datasets, and has a large body of agricultural-domain precedent to build on and compare against, versus e.g. hyperspectral pipelines which have far less mature open tooling.

## 8. Challenges in real-world deployment

This is where most published work stops short, and where a proposal's differentiation should live, since deployment-ready inference pipelines are explicitly required:

- **Connectivity constraints.** Rural field locations often have poor or no internet access, so cloud-inference-only architectures are frequently non-viable — favors on-device (edge) inference or hybrid online/offline design.
- **Compute constraints on-device.** Deployable models must run on low/mid-range smartphones or edge devices, which pushes toward MobileNet/EfficientNet-Lite-class architectures, quantization, and pruning rather than the largest-accuracy model in isolation.
- **Environmental variability at capture time.** Lighting, occlusion, background clutter, multiple co-occurring symptoms on one leaf, and non-standardized camera angles by non-expert users at the point of capture all degrade real accuracy versus benchmark accuracy — precisely the domain-shift problem from point 6, but now as an engineering constraint rather than a research curiosity.
- **Trust and adoption.** A model that gives no explanation or confidence calibration is unlikely to change farmer behavior (e.g., trigger a pesticide purchase) — interpretability and confidence thresholds for "refer to human expert" fallback paths matter operationally, not just academically.
- **Maintenance drift.** Disease presentation, pest ranges, and even crop varieties shift over seasons/years (climate-driven, per point 2), so a static model degrades — exactly why continuous improvement based on new ground-truth data belongs in the proposal as an explicit deliverable, not an afterthought.

## 9. Expected future developments

Recent literature (2025–2026) points toward a few converging directions worth positioning this proposal near, without overclaiming them as this project's own contribution:

- **Agriculture-specific foundation models and few-shot/zero-shot adaptation** — reducing dependence on large labelled datasets per new disease class, relevant since any provided dataset, however good, will still have long-tail rare classes.
- **Hybrid CNN–Transformer and lightweight ViT variants (e.g., GreenViT-class architectures)** — balancing the accuracy advantages of attention mechanisms with the compute budget needed for edge deployment.
- **Explainable AI as a standard reporting requirement**, not an add-on, especially for field-facing agricultural tools where trust drives adoption.
- **Field-native benchmark datasets** (e.g., CCMT-style datasets with natural backgrounds and co-occurring symptoms) increasingly replacing PlantVillage-only evaluation as the credible standard — meaning the comparative evaluation methodology should explicitly test on field-like conditions, not just report accuracy on the training distribution, if it's to be taken seriously by reviewers already familiar with this critique.

---

This review grounds the Background, Statement of the Problem, and Objectives sections of the proposal, and feeds directly into the crop-selection analysis and model-comparison documents that follow.
