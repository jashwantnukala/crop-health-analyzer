# Deployment Strategy — Turmeric & Ridge Gourd Disease Detection
### TIH-IoT IIT Bombay CHANAKYA Fellowship 2026

This document specifies the deployment strategy referenced as Stage 10 in the technical architecture, expanded to the level of detail the CFP's "deployment-ready inference pipeline for operational use" deliverable requires. It covers every deployment surface (API, desktop, cloud, edge, future mobile), the model-optimization techniques that make edge deployment viable (ONNX, TensorRT, quantization, compression), and the operational layer (Docker, monitoring, logging) that keeps the system trustworthy after the fellowship ends. Every choice is justified against two things: the field-connectivity constraint established in the literature review, and the model characteristics produced in Stage 8's comparison (Section 4).

---

## 1. Design principle: deployment target dictates optimization path, not the reverse

A common mistake in academic CV projects is training one model and then asking "how do we deploy this." That ordering breaks down here because the four-to-five candidate models compared in Stage 8 (MobileNetV3, EfficientNet-B0, ResNet-50, ViT-B/16, hybrid CNN-ViT) have deployment costs that differ by more than an order of magnitude in parameter count and latency. The correct ordering, and the one this proposal follows, is:

1. Identify the deployment surfaces the CFP and the target user (farmer / extension worker / TIH field office) actually need.
2. Determine the compute/connectivity envelope of each surface.
3. Select — from the already-trained and compared candidates — which model(s) fit which envelope, applying compression/quantization where needed to fit a model that wouldn't otherwise fit.
4. Only then build the serving infrastructure around that decision.

This is why the deployment strategy is written after model comparison (Stage 8) and not before — deployment feasibility is a *selection criterion* for the final model, not an afterthought bolted onto whichever model scored highest on accuracy alone.

---

## 2. Inference Pipeline (deployment-agnostic core)

**Purpose:** A single, versioned inference module shared by every serving surface below, so the API, desktop app, and edge binary never diverge in preprocessing/postprocessing logic — a common and avoidable source of train-serve skew.

**Stages (all four deployment surfaces call this same pipeline):**

```
Image input → Validation (format/size/corruption check)
           → Preprocessing (resize, colour-normalize, ROI-crop — identical to Stage 2 of training pipeline)
           → Model forward pass (backend-specific: PyTorch / ONNX Runtime / TensorRT, see Section 5)
           → Postprocessing (softmax → calibrated confidence, Section 9's temperature scaling)
           → Explainability (Grad-CAM overlay, computed only above a configurable confidence threshold to control latency cost)
           → Structured output: {disease_class, confidence, PDI_estimate (if E2 multi-task head shipped), heatmap, model_version, inference_time_ms}
```

**Why this matters as its own deliverable:** preprocessing mismatch between training and serving (e.g., different resize interpolation, different normalization constants) is one of the most common causes of a model performing well in evaluation but poorly in production, and it is silent — it does not throw an error, it just degrades accuracy. Encoding preprocessing as a single versioned, testable module (packaged with the model artifact, not reimplemented per platform) is a deliberate defence against this failure mode, not boilerplate.

**Implementation:** a thin Python package (`inference_core`) with backend-agnostic pre/post-processing and a pluggable model-runner interface, so swapping the backend (PyTorch → ONNX → TensorRT) touches one class, not every deployment surface.

---

## 3. Model Compression and Quantization

These are treated together because in this project they are sequential steps on the same model, not competing approaches.

### 3.1 Model Compression

| Technique | What it does | Applicability here |
|---|---|---|
| **Pruning** (structured, channel-level) | Removes redundant filters/channels below an importance threshold | Applicable to MobileNetV3/EfficientNet-B0's convolutional layers; structured pruning is preferred over unstructured (weight-level) pruning because it produces genuinely smaller, faster-to-run tensors — unstructured pruning creates sparse matrices that most mobile/edge runtimes cannot exploit for real speedup, only real storage savings, which is the wrong trade-off for a latency-sensitive field application |
| **Knowledge distillation** | Trains a small "student" model to match a larger "teacher" model's output distribution, not just hard labels | Genuinely useful here: distill the best-performing heavier candidate (ResNet-50 or the hybrid CNN-ViT from Stage 8) into MobileNetV3-sized student, transferring some of the accuracy advantage of the heavier model into a deployable footprint. This is the compression technique with the best accuracy-retention track record for this project's accuracy gap, and is proposed as the primary compression step, not pruning |
| **Low-rank factorization** | Decomposes weight matrices into lower-rank approximations | Lower priority — smaller, less mature tooling support for CNN backbones compared to distillation/pruning, and marginal gains once MobileNetV3 is already the base architecture (it's already an efficiency-optimized design, so headroom is smaller) |

**Recommendation:** knowledge distillation (heavy model → MobileNetV3 student) as the primary compression technique, since it directly targets this project's actual trade-off (heavier models measured more accurate in Stage 8, but not deployable at the edge as-is), with structured pruning as a secondary, smaller-gain optimization applied after distillation if further size reduction is needed for low-end Android devices.

### 3.2 Quantization

| Technique | Precision | Accuracy impact | When used here |
|---|---|---|---|
| **Post-Training Quantization (PTQ), INT8** | FP32 → INT8, no retraining | Typically 0.5–2% accuracy drop for CNNs, larger and less predictable for ViT-family models due to their sensitivity to attention-score quantization | Default choice for the edge-deployed MobileNetV3/EfficientNet-B0 model — cheapest to apply, and the accuracy drop is measurable and boundable via the calibration step below |
| **Quantization-Aware Training (QAT)** | Simulates INT8 quantization during fine-tuning | Recovers most of PTQ's accuracy loss, typically <0.5% drop | Used only if PTQ's measured accuracy drop (validated on the held-out field-validation set, not just the lab test set) exceeds an acceptable threshold — QAT costs additional training time and is applied selectively, not by default, to stay within the 10-month budget |
| **FP16 (half-precision)** | FP32 → FP16 | Negligible accuracy drop | Used for the cloud/API-served heavier models (ResNet-50, ViT-B/16) on GPU inference — halves memory and improves throughput with essentially no accuracy cost, unlike INT8 which is riskier for transformer attention layers |

**Critical discipline point:** every quantized model is re-evaluated on the *field-validation* set (Stage 6b), not just the lab test set, before being shipped — a model can pass INT8 quantization cleanly on clean lab images while its accuracy drop is amplified on noisier field images, and this would only be caught by re-testing on the harder set.

**Recommendation:** INT8 PTQ as the default for the on-device model, with a documented fallback to QAT if the field-validation accuracy drop exceeds a pre-agreed threshold (proposed: 2 percentage points relative to the FP32 baseline) — this makes the quantization decision evidence-based and pre-committed rather than an ad hoc choice made under deadline pressure.

---

## 4. ONNX

**What it is:** an open, framework-neutral model interchange format. A model trained in PyTorch is exported once to ONNX and can then be run by ONNX Runtime on CPU, GPU, mobile (ONNX Runtime Mobile), or Windows/desktop, without depending on a full PyTorch installation at inference time.

**Why it is used here, not just PyTorch everywhere:**
- **Portability across deployment surfaces.** The same `.onnx` artifact serves the desktop app (Section 6), the offline edge path (Section 7), and can act as an intermediate step toward TensorRT (Section 5) — one export step, three consumption paths, rather than maintaining separate serialized models per platform.
- **Smaller runtime footprint.** ONNX Runtime's binary is substantially lighter than a full PyTorch install, which matters directly for the desktop application's install size and the Android future-work path's app size (Section 8).
- **Graph-level optimizations applied at export** (operator fusion, constant folding) — these are close to free accuracy-neutral speedups obtained automatically by the ONNX export/optimization tooling, not something the team has to hand-engineer.

**Where it is not the final answer:** ONNX Runtime is CPU/general-GPU inference — it is not itself a maximally-optimized NVIDIA-GPU inference engine. That is what TensorRT is for.

---

## 5. TensorRT

**What it is:** NVIDIA's inference optimization SDK and runtime, which takes a trained model (commonly via ONNX as the intermediate format) and produces a hardware-specific, highly optimized execution engine for NVIDIA GPUs — applying kernel auto-tuning, layer fusion, and precision calibration (FP16/INT8) specific to the target GPU.

**Why (and where) it is used here:** TensorRT is proposed **only for the cloud/API-serving path** (Section 6.4), specifically for the heavier candidate models (ResNet-50, ViT-B/16, hybrid CNN-ViT) when served on GPU-backed infrastructure — TensorRT's typical 2–5× latency improvement over unoptimized GPU inference directly reduces the per-request cost and latency of the API path, which matters for a service intended to handle concurrent requests from multiple TIH field offices.

**Why it is *not* proposed for the on-device/edge path:** TensorRT targets NVIDIA GPUs specifically. It has no role in the offline, farmer-facing Android/edge path, which targets ARM mobile CPUs/NPUs — that path correctly uses ONNX Runtime Mobile / TFLite instead (Section 7). Proposing TensorRT for edge deployment would be a category error given the actual target hardware, and this proposal deliberately avoids listing it as a universal solution to signal that the choice is hardware-aware, not a checklist of trendy tools.

**Pipeline:** PyTorch model → export to ONNX → ONNX → TensorRT engine build (with INT8 calibration using a representative field-image calibration set, not random data, since calibration-set mismatch is a known cause of TensorRT INT8 accuracy loss) → serve behind the FastAPI layer.

---

## 6. Serving Surfaces

### 6.1 REST API

**Purpose:** the primary interface for any client with connectivity — TIH field-office kiosks, extension-worker tablets, the desktop app, and (future) mobile app's online mode.

**Design:**
- **Framework: FastAPI**, chosen over Flask/Django for this specific use case because of native async support (relevant when Grad-CAM computation or batched inference adds latency), automatic OpenAPI schema generation (useful documentation deliverable for TIH-IoT's "future maintenance" requirement at near-zero extra effort), and built-in request validation via Pydantic (catches malformed requests before they reach the model, rather than failing inside inference code).
- **Endpoints:** `/predict` (single image), `/predict/batch` (multiple images — relevant for a field officer surveying multiple plants in one visit), `/health` (liveness/readiness for the monitoring stage), `/models` (lists currently deployed model version(s) — supports the versioned-model-registry requirement from the technical architecture document).
- **Model backend:** ONNX Runtime by default; TensorRT engine when GPU infrastructure is available (Section 5), selected via a config flag so the same API code serves either backend — this keeps the "which backend" decision an infrastructure choice, not a code fork.
- **Versioning:** every response includes the serving model's version tag, tying every prediction back to a specific entry in the model registry (technical architecture, Stage 10) — required for the retraining-trigger traceability the monitoring stage depends on.

### 6.2 Desktop Application

**Purpose:** targeted specifically at TIH-IoT field offices and extension-worker stations with an available laptop/desktop but potentially unreliable or intermittent connectivity — a middle ground between the always-online API and the phone-only edge path.

**Comparison of build approaches:**

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| **Electron + ONNX Runtime Web/Node bindings** | Familiar web-stack UI development; cross-platform (Windows/Linux) with one codebase | Large install footprint (Chromium bundled); higher memory usage — a real concern on the lower-spec machines likely available at rural field offices | Not recommended given the target hardware profile |
| **Python (PyQt/Tkinter) + ONNX Runtime (native)** | Directly reuses the same `inference_core` module and ONNX artifact used by the API and edge paths (Section 2) — no reimplementation; lighter footprint than Electron; simplest to maintain for a small fellowship team, since it stays in one language (Python) across API, desktop, and training code | Less polished UI out-of-the-box than a web-stack app; Windows packaging (PyInstaller) needs care to keep install size reasonable | **Recommended** |
| **Native (C++/Qt)** | Best performance, smallest footprint | Disproportionate development cost for a 10-month fellowship with one core team building the ML system, not a dedicated desktop-app engineering track | Not recommended — cost does not match project scope |

**Recommendation and justification:** Python + PyQt, running the same ONNX model and `inference_core` pipeline as the API, packaged as a standalone executable (PyInstaller) with the quantized on-device model bundled for full offline operation — giving the desktop app the API's UI convenience and the edge path's offline reliability, at a development cost proportionate to a 10-month UG fellowship team. This is a genuine engineering trade-off decision, not a default choice: the alternative (Electron) is more common in industry generally but is the wrong choice specifically given this project's target hardware and team size.

### 6.3 Edge Deployment (offline, farmer-facing)

Already specified at architecture level in the technical architecture document (Stage 10); this section adds the deployment-technique detail.

**Target:** Android devices with ONNX Runtime Mobile (preferred over TFLite here — see comparison below), running the compressed + quantized MobileNetV3/EfficientNet-B0 model from Section 3.

**ONNX Runtime Mobile vs. TFLite:**

| Criterion | ONNX Runtime Mobile | TFLite |
|---|---|---|
| Framework lock-in | Framework-neutral — same export pipeline as the API/desktop paths (Section 4) | Requires a separate TensorFlow-format conversion path if training was done in PyTorch, adding a second export/validation pipeline to maintain |
| Tooling maturity for CNN mobile deployment | Mature, actively developed by Microsoft, good INT8 support | Very mature, historically the default choice for Android on-device ML, marginally broader Android-specific tooling/community examples |
| Consistency with rest of this project | High — one export format (ONNX) serves API, desktop, and edge | Would require maintaining ONNX for API/desktop and a separate TFLite export for mobile — real, avoidable engineering overhead for a small team |

**Recommendation:** ONNX Runtime Mobile, specifically to preserve the single-export-format discipline established in Section 4 — TFLite is a reasonable alternative in isolation, but is the wrong choice *for this project* given it would fragment the export pipeline the rest of the architecture is built around.

**Operational behaviour:** predictions are computed fully on-device (no network call required), logged locally, and synced to the monitoring backend when connectivity resumes — exactly the offline/online branching already specified in the technical architecture's Stage 10 diagram.

### 6.4 Cloud Deployment

**Purpose:** hosts the REST API (Section 6.1) and, where GPU resources are available, the TensorRT-optimized heavier models — serving TIH field offices, the desktop app's online mode, and (future) the mobile app's online mode.

**Containerization: Docker.** The API, its ONNX Runtime/TensorRT backend, and all dependencies are packaged as a Docker image, for three concrete reasons specific to this project rather than "Docker is standard practice":
1. **Reproducible handover.** The CFP's "future model maintenance" requirement means the system must run identically for a TIH-IoT team member after the fellowship ends, on infrastructure the current team doesn't control — a Docker image removes "works on my machine" as a failure mode for that handover.
2. **Environment isolation between the CUDA/TensorRT stack (GPU serving) and the CPU-only ONNX Runtime stack (lighter serving)** — these are proposed as two separate container images sharing the same `inference_core` codebase, so a deployment without GPU access (a real possibility for a TIH field office) can still run the CPU-served path without CUDA/TensorRT dependency conflicts.
3. **Straightforward scaling** — container orchestration (a managed Kubernetes service, or simpler container-hosting if request volume doesn't justify Kubernetes's operational overhead — a judgment call to be made against actual measured request volume, not assumed upfront) allows horizontal scaling of the API path if concurrent field-office usage grows.

**Cloud provider positioning:** this proposal deliberately does not commit to a specific cloud vendor at this stage — the containerized design means the choice (AWS/GCP/Azure/on an IIT-B-hosted server) is an infrastructure decision to be made with TIH-IoT based on their existing infrastructure and budget, not a design constraint on the ML system itself. This is stated explicitly because over-committing to a vendor in a research proposal is a common and avoidable weakness reviewers notice.

---

## 7. Monitoring

Expanding on the technical architecture's Stage 11 with deployment-surface-specific detail:

- **Per-surface health metrics:** API uptime/latency/error rate (standard APM, e.g. Prometheus + Grafana — open-source, self-hostable, consistent with not over-committing to a paid vendor); on-device crash rate and inference latency reported on sync (edge/desktop paths); Docker container resource utilization (CPU/GPU/memory) for the cloud path.
- **Model-quality metrics** (data drift, confidence drift, expert-disagreement rate) as already specified in the technical architecture — reiterated here only to note that these are computed centrally regardless of which surface generated the prediction, since offline edge predictions sync into the same monitoring backend as online API predictions (per the Stage 10 architecture diagram). A drift signal from edge-collected images is just as actionable as one from the API, and treating them separately would create a blind spot for exactly the farmer-facing usage the project is meant to serve.

## 8. Logging

- **Structured, not free-text.** Every inference call (across all four surfaces) emits a structured log record — `{timestamp, model_version, surface (api/desktop/edge/mobile-future), input_hash, prediction, confidence, latency_ms}` — as JSON, not a printed string, because structured logs are what the monitoring stage's drift computations and the retraining-trigger policy (technical architecture, Stage 11) actually consume programmatically.
- **Privacy-aware image handling.** Raw farmer-submitted images are not logged verbatim by default — only a hash/thumbnail plus prediction metadata — with full-image retention opt-in and tied to the field-validation feedback loop specifically, not blanket collection. This is a deliberate, statable design choice for a system handling images potentially tied to identifiable farms/locations, and worth stating explicitly in the proposal as evidence of research maturity around data governance, not just model accuracy.
- **Centralized aggregation.** Desktop and edge logs are queued locally (already specified in Stage 10's offline-sync behaviour) and shipped to the same centralized log store as API logs on sync, so the monitoring dashboards in Section 7 have one consistent view across all deployment surfaces rather than four disconnected logging systems.

---

## 9. Future Mobile Deployment (explicitly scoped as future work, not a 10-month deliverable)

Given the 10-month fellowship timeline, a polished, published Android/iOS app is realistically **future work**, not a Month-10 deliverable — this is stated explicitly rather than implied, since overcommitting the mobile-app scope is a common weakness in similar proposals that reviewers are likely to notice.

**What the 10-month project *does* deliver toward this goal** (making the "future deployment" claim credible rather than a throwaway line):
- The ONNX-exported, INT8-quantized, compressed model (Section 3, Section 4) is *already* in the correct format for ONNX Runtime Mobile integration — the model-readiness work is complete within the 10 months, even if the polished consumer app UI is not.
- The `inference_core` pipeline (Section 2) is platform-agnostic by design, so the pre/postprocessing logic ports directly into an Android app's inference code without reimplementation.
- The offline/online sync behaviour and structured logging (Sections 6.3, 8) are already designed around a mobile-first offline-capable client, so a future mobile app is integrating into an existing architecture, not requiring new backend design.

**Recommended future path:** Android-first (given the CFP's rural-India target user base and Android's dominant market share in that segment, versus iOS), using ONNX Runtime Mobile with the same model artifact validated in this project — proposed as a natural Phase 2 extension for TIH-IoT to pursue post-fellowship, or as a stretch goal in the final 2–4 weeks if the core deliverables are complete ahead of schedule.

---

## 10. Recommended Deployment Architecture — summary and justification

**Recommendation:** a **hybrid, format-unified architecture** —

- **One export format (ONNX)** feeding all serving surfaces, avoiding the fragmentation of maintaining separate PyTorch/TFLite/native pipelines.
- **On-device inference (ONNX Runtime Mobile, INT8-quantized, distilled MobileNetV3/EfficientNet-B0)** as the primary, farmer-facing path — because the literature review's connectivity-constraint finding makes an always-online architecture unrealistic for the actual target user.
- **Containerized REST API (FastAPI + ONNX Runtime, with a TensorRT-optimized path for GPU-backed heavier models)** for TIH field offices and extension workers with reliable connectivity, serving both the higher-accuracy heavier candidates and as the sync target for offline predictions.
- **A Python-based desktop application** reusing the identical inference core, for field offices with intermittent connectivity — filling the gap between phone-only edge and always-online API.
- **Docker-containerized, vendor-neutral cloud hosting**, positioned as an infrastructure decision for TIH-IoT rather than a fixed proposal commitment.
- **Centralized, structured monitoring and logging** across all four surfaces, operationalizing the CFP's continuous-improvement deliverable rather than treating it as aspirational language.
- **Mobile app explicitly scoped as future work**, with the current architecture deliberately designed so that future work is a natural extension, not a redesign.

**Why this architecture over the two obvious simpler alternatives:**
- *"Just build a cloud API"* fails the literature review's own connectivity-constraint finding — it would not actually be usable by the target farmer in a low-connectivity field setting, undermining the project's stated real-world impact.
- *"Just build an on-device app, skip the API"* fails the CFP's requirement to also serve TIH field offices/extension workers, and forecloses running the heavier, more accurate models (ResNet-50, ViT-B/16, hybrid) that Stage 8's comparison found more accurate but too costly for edge deployment — discarding that comparative-evaluation work rather than using it.

The hybrid design is the direct engineering answer to *both* constraints simultaneously, using the same underlying model-comparison and optimization work (Stages 8–9) rather than requiring a second, separate deployment-specific research effort — which is why it is proposed as a coherent architecture rather than a checklist of deployment technologies.
