# Crop Selection Analysis — Image-Based Crop Disease Detection
### Crops Considered: 30 TIH-IoT Candidate Crops → Turmeric & Ridge Gourd
### TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This document specifies how two crops were selected from the 30 candidates offered under the TIH-IoT dataset, using a nine-axis scoring framework grounded in the literature review, and closes with a recommended pairing justified against the fellowship's comparative-modelling requirements.

---

## 0. Scoring framework

Before ranking, it's necessary to define what "good" means on each axis, because these criteria pull in different directions and a naive "pick the most-researched crop" or "pick the most novel crop" both fail for different reasons.

| Criterion | What high score means here |
|---|---|
| Agricultural importance | Area under cultivation, farmer dependency, export/GI value in India |
| Disease complexity | Multiple disease agents (fungal/bacterial/viral/pest), overlapping visual symptoms — genuinely justifies comparing 3–5 models rather than one being trivially sufficient |
| Dataset suitability | Existence of public benchmark/pretraining data to validate methodology against, even though the TIH dataset supplies the primary labelled set |
| Research novelty | Low saturation — if 200 papers already report >99% accuracy on this crop, a 10-month UG project adds little |
| Deployment feasibility | Symptoms visible via ordinary smartphone RGB imagery, not requiring hyperspectral/lab equipment |
| Social impact | Smallholder relevance — who actually benefits if this ships |
| AI difficulty | Fine-grained/subtle symptom differentiation that makes model comparison scientifically meaningful, not trivially easy |
| Expected evaluation metrics | Enough disease classes for meaningful precision/recall/F1 breakdown, not just binary healthy/diseased |
| Prior research availability | Enough literature to benchmark against and cite methodologically, without being so saturated it's derivative |

Each criterion is scored 1 (weak) – 5 (strong), based on the literature review plus established agronomic knowledge. Where a score was inferred rather than sourced from a located paper, it is marked *(inferred)* and should be treated as a reasonable prior to verify, not settled fact, when the literature section is finalized.

## 1. Full ranking — all 30 crops

| Crop | Agri. Importance | Disease Complexity | Dataset Suitability | Research Novelty | Deployment Feasibility | Social Impact | AI Difficulty | Eval. Metric Richness | Prior Research | **Total /45** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Turmeric** | 4 | 5 (foliar + rhizome) | 3 | 4 | 4 | 5 (GI-tag export crop) | 4 | 4 | 3 | **36** |
| **Ridge Gourd** | 3 | 4 (viral YMD dominant, *ToLCNDV*) | 2 *(inferred)* | 5 (almost no CV work found) | 4 | 4 | 4 | 3 | 2 | **31** |
| **Ginger** | 3 | 4 (bacterial wilt, multiple fungal) | 3 | 4 | 4 | 4 | 4 | 4 | 3 | **33** |
| **Okra** | 4 | 3 *(inferred)* | 3 *(inferred)* | 4 | 4 | 4 | 3 | 3 | 3 *(inferred)* | **31** |
| **Bitter Gourd** | 3 | 3 *(inferred, viral+fungal)* | 2 *(inferred)* | 4 *(inferred)* | 4 | 4 | 3 | 3 | 2 *(inferred)* | **28** |
| **Capsicum** | 4 | 4 (bacterial spot, viral, fungal) | 3 *(inferred)* | 3 | 4 | 4 | 4 | 4 | 3 *(inferred)* | **33** |
| **Cucumber** | 3 | 3 | 4 (PlantVillage-adjacent) | 2 (heavily studied) | 4 | 3 | 3 | 3 | 5 | **30** |
| **Cabbage** | 3 | 3 | 3 *(inferred)* | 3 | 4 | 3 | 3 | 3 | 3 *(inferred)* | **28** |
| **Cauliflower** | 3 | 3 | 3 *(inferred)* | 3 | 4 | 3 | 3 | 3 | 3 *(inferred)* | **28** |
| **Carrot** | 3 | 2 *(inferred)* | 3 *(inferred)* | 3 | 3 (root crop, foliage-only signal) | 3 | 3 | 3 | 3 *(inferred)* | **26** |
| **Watermelon** | 3 | 3 *(inferred)* | 3 *(inferred)* | 3 | 4 | 3 | 3 | 3 | 3 *(inferred)* | **28** |
| **Muskmelon** | 2 | 3 *(inferred)* | 2 *(inferred)* | 4 | 4 | 2 | 3 | 3 | 2 *(inferred)* | **25** |
| **Custard Apple** | 3 | 4 (6-class documented dataset exists) | 4 | 3 | 4 | 3 | 4 | 4 | 3 | **32** |
| **Guava** | 3 | 3 | 4 (multiple published datasets, GLD-Det etc.) | 2 (fairly saturated already) | 4 | 3 | 3 | 3 | 4 | **29** |
| **Sapota (Chikoo)** | 2 | 3 *(inferred)* | 1 *(inferred, almost no public data)* | 5 | 4 | 2 | 3 | 2 | 1 *(inferred)* | **23** |
| **Dragon Fruit** | 2 | 3 (stem canker, bacterial, sunburn) | 3 (recent 2025 datasets exist) | 4 | 4 | 2 (niche, high-value crop, low smallholder reach) | 3 | 3 | 3 | **27** |
| **Papaya** | 3 | 3 | 4 | 2 (well studied incl. smartphone datasets) | 4 | 3 | 3 | 3 | 4 | **29** |
| **Citrus** | 4 | 4 (huanglongbing/citrus greening — globally critical, hard problem) | 4 (large public datasets exist) | 2 (very well studied globally) | 4 | 4 | 4 | 4 | 5 | **35** |
| **Cashew Nut** | 3 | 3 (anthracnose, dieback) | 2 *(inferred)* | 4 | 3 (canopy/nut imaging harder than leaf) | 3 | 3 | 3 | 2 *(inferred)* | **26** |
| **Strawberry** | 2 | 3 | 5 (part of PlantVillage/well-benchmarked) | 1 (extremely saturated) | 4 | 2 | 2 | 3 | 5 | **27** |
| **Black Gram (Urad)** | 4 | 3 (Yellow Mosaic Virus dominant) | 2 *(inferred)* | 4 *(inferred)* | 4 | 4 | 3 | 3 | 2 *(inferred)* | **29** |
| **Green Gram (Moong)** | 4 | 3 (Yellow Mosaic Virus dominant) | 2 *(inferred)* | 4 *(inferred)* | 4 | 4 | 3 | 3 | 2 *(inferred)* | **29** |
| **Cowpea** | 3 | 3 *(inferred)* | 2 *(inferred)* | 4 *(inferred)* | 4 | 3 | 3 | 3 | 2 *(inferred)* | **27** |
| **Kidney Bean (Rajma)** | 3 | 3 *(inferred, rust/anthracnose)* | 2 *(inferred)* | 4 *(inferred)* | 4 | 3 | 3 | 3 | 2 *(inferred)* | **27** |
| **Masur (Lentil)** | 3 | 2 *(inferred)* | 2 *(inferred)* | 4 *(inferred)* | 4 | 3 | 2 | 2 | 2 *(inferred)* | **24** |
| **Pea** | 3 | 3 (powdery mildew common, well characterized) | 3 *(inferred)* | 3 | 4 | 3 | 3 | 3 | 3 *(inferred)* | **28** |
| **Cluster Bean (Guar)** | 3 (industrial gum crop, arid-zone importance) | 2 *(inferred)* | 1 *(inferred)* | 5 (almost no CV literature) | 4 | 3 | 2 | 2 | 1 *(inferred)* | **23** |
| **Bajra (Pearl Millet)** | 4 (major arid-zone staple) | 3 (blast, downy mildew, ergot) | 2 *(inferred)* | 4 *(inferred)* | 3 (grain/panicle symptoms harder than broadleaf) | 5 (dryland smallholder staple) | 4 | 3 | 2 *(inferred)* | **30** |
| **Sunflower** | 3 | 3 (rust, downy mildew) | 3 *(inferred)* | 3 | 4 | 3 | 3 | 3 | 3 *(inferred)* | **28** |
| **Safflower** | 2 | 2 *(inferred)* | 1 *(inferred)* | 5 (almost no CV literature) | 3 | 2 | 2 | 2 | 1 *(inferred)* | **20** |

## 2. Top 5, ranked, with reasoning

**1. Turmeric — 36/45.** This is the strongest candidate on the full profile, not just the total score. Turmeric disease has a structural property almost no other crop on this list shares: the economically damaging disease (rhizome rot) occurs *underground*, while the diagnostic signal a camera can actually capture is the *foliar* symptom (leaf blotch, dry leaf). That's a genuinely hard, genuinely novel computer-vision framing — this is not just classifying a lesion, but modelling a foliage-to-rhizome disease correlation, which very few existing papers address (the one dataset paper found treats them as separate classes, not a joint inference problem). Combined with turmeric's outsized role in Indian export agriculture (India produces the large majority of the world's turmeric, much of it GI-tagged, smallholder-grown), this gives both scientific novelty and a strong social-impact narrative for the proposal.

**2. Citrus — 35/45.** Citrus scores extremely high on agricultural importance, disease severity (Huanglongbing/citrus greening is one of the most economically destructive plant diseases globally), and dataset maturity. The problem is research novelty (2/5) — citrus greening detection is one of the most heavily published CV-agriculture problems in the world; a 10-month UG project is unlikely to say anything reviewers haven't already seen many times. Strong crop, weak choice for this fellowship's novelty bar.

**3. Ginger — 33/45.** Close structural cousin to turmeric (rhizome spice, similar disease biology — bacterial wilt plus multiple fungal pathogens), decent existing baseline literature to benchmark against, strong deployment feasibility, high social relevance. A very defensible second pick, and arguably the safer pairing partner to turmeric since the two share methodology (rhizome-crop imaging pipelines, similar preprocessing) — but that similarity is also a weakness: pairing two rhizome spice crops narrows the methodological range showcased in the comparative-evaluation-across-crops story the fellowship asks for.

**4. Capsicum — 33/45.** Strong all-around vegetable candidate: bacterial spot, viral mosaic, and fungal disease all present, meaning real multi-etiology complexity; high deployment feasibility; solid social relevance (major protected-cultivation and open-field crop in India). Reasonable novelty since it's less saturated than tomato/potato-style PlantVillage staples.

**5. Custard Apple — 32/45.** Genuinely interesting: a documented 6-class fruit-and-leaf disease dataset already exists (8,226 images, Indian-grown), giving methodological grounding, but the crop is still under-published enough that a comparative model study would be a real contribution rather than incremental. Weaker than turmeric on social impact/scale (custard apple is a valuable but comparatively niche horticultural crop, not a staple).

## 3. Recommended pair: Turmeric + Ridge Gourd

The recommendation steers away from pairing turmeric with ginger or citrus, and toward **Turmeric + Ridge Gourd**, for a specific methodological reason, not just because their individual scores are high.

**Why this pair, scientifically:**

1. **Disease-etiology diversity, which is the actual point of a comparative-modelling project.** Turmeric's dominant disease burden is fungal/foliar-vs-rhizome (leaf blotch, rhizome rot — texture- and color-degradation symptoms, roughly continuous/gradual visual progression). Ridge gourd's dominant disease burden is viral (Yellow Mosaic Disease caused by *ToLCNDV*, a *Begomovirus* transmitted by whitefly) — mottling, chlorosis, and leaf distortion patterns that are visually and pathologically distinct from fungal lesions. If a proposal built and compared CNN/ViT architectures across only similar disease types (e.g., turmeric + ginger, both rhizome-fungal), a reviewer could reasonably ask "did you actually test generalizability, or just get lucky on one symptom class?" Pairing a fungal/structural-degradation crop with a viral/color-pattern crop allows a real claim about which architectures handle which symptom type better — a legitimate, citable research question (fine-grained texture vs. color-pattern discrimination is a known CV distinction), not just proposal padding.

2. **Ridge gourd's ToLCNDV problem is a genuinely underserved research gap.** There is essentially no computer-vision-specific literature on ridge gourd disease detection — the only relevant source found was an antibody-based (serological) diagnostic method, not imaging. That's unusually low saturation for a widely-grown Indian vegetable, meaning a comparative model study here would be closer to first-of-its-kind for this crop rather than an incremental improvement on an already-solved benchmark — a stronger claim for a fellowship review panel than "re-ran ResNet on tomato."

3. **Deployment story is coherent across both.** Both are smallholder, open-field crops (not greenhouse/protected cultivation like some capsicum production), both present disease symptoms clearly on standard leaf/vine imagery obtainable via smartphone, and both have real-world urgency — ridge gourd YMD is explicitly flagged in the literature as an emerging major problem for the crop in India, meaning a working detection tool has genuine field utility, not just academic interest.

4. **Complementary evaluation-metric profiles.** Turmeric gives a genuinely multi-class problem with class imbalance considerations (multiple disease classes at different prevalence) — good grounds for reporting per-class F1/precision-recall rather than aggregate accuracy, the correct rigor per the research gaps identified in the literature review. Ridge gourd, with a more dominant single-disease (YMD) burden, allows a cleaner binary/few-class severity-grading pipeline (e.g., healthy / early infection / severe infection) — a different but equally valid evaluation design. Handling both problem shapes is a stronger technical showcase than two crops with near-identical class structures.

**One honest caveat to flag to the faculty mentor before finalizing:** ridge gourd scored a 2/5 on "prior research availability" specifically because no CV-specific prior work could be found to benchmark against — that cuts both ways. It's real novelty, but it also means the literature review section for ridge gourd will be thinner, and the team will be doing more first-principles methodology design (less "here's what worked for others") rather than replicate-and-improve. If the team's risk tolerance or timeline is tight, **Turmeric + Capsicum** is the safer high-scoring alternative with a similarly complementary disease-type story (rhizome/foliar-fungal vs. multi-etiology fungal-bacterial-viral) and a bit more existing literature to lean on.

---

This crop selection feeds directly into the literature review's crop-specific pass, the model-design-and-comparison document, and the technical architecture document, all built around the confirmed pair of Turmeric and Ridge Gourd.
