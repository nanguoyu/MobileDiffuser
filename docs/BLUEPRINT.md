# Universal Diffusion App — Rebuild Blueprint

> A universal (macOS + iOS) Swift app for on-device image generation, built entirely on
> **Apple MLX + Swift**, with first-class model management, smart per-hardware
> load/unload, and support for many open-weight diffusion models at multiple precisions —
> all downloadable in-app.

This document is the canonical plan for rebuilding this repository. It supersedes the
original CoreML iPhone app: the UI and logic are torn down and rebuilt from scratch.

---

## 1. North star

- **One engine, everywhere.** A single **MLX/Swift** inference stack runs on both Mac and
  iPhone. No CoreML. The cost (MLX on a phone is heavier/slower than ANE) is accepted in
  exchange for one elegant, unified codebase and the freedom to add any open MLX model.
- **Partial load is how big models fit a phone.** Rather than a second engine, we make the
  one engine memory-frugal through a streaming/partial-load ladder (below).
- **Model management is the product.** Picking a model + precision, seeing whether it fits
  *this* device, downloading it (resumably, from public sources or an external SSD), and
  running it — that whole loop is the core experience.
- **Open source, open weights.** Only public, openly-licensed models and dependencies.

---

## 2. Engine strategy — pure MLX, made to fit via partial load

MLX-on-GPU uses more peak memory than ANE for the same model, but the gap is mostly
*optimization effort* (4-bit quantization + per-submodule streaming), not an inherent
framework penalty. We close it with a ladder the engine climbs only as far as a given
model/device requires:

| Rung | Technique | Effect | Prior art |
|---|---|---|---|
| 1 | 4-bit (down to 2/3-bit) quantization | halve, then halve again | community MLX weights |
| 2 | **Two-phase staging** — load text encoder → encode → release → load transformer + VAE → denoise → decode | encoder and transformer never co-reside | mflux `--low-ram` |
| 3 | **Block streaming** — load each denoiser block from mmap'd / ranged-read weights, run, release | transformer residency drops from GB to hundreds of MB | SwiftLM SSD streaming on mlx-swift; split-stage CoreML |
| 4 | Memory-efficient attention (flash / chunked SDPA) | bounds activation memory at high resolution | MLX `scaledDotProductAttention` |
| 5 | Device gating — `MemoryProbe.availableBytes()` + `MLX.GPU.set(cacheLimit:)` per device | runtime resident-vs-stream decision | jetsam-accurate probe |

### Feasibility (verified sizes, two-phase peak ≈ max(encoder, transformer+VAE) + working set)

| Model (MLX) | Disk | Peak (Q4) | 8 GB iPhone (~4 GB budget) | 12 GB iPhone (~6 GB) | Mac |
|---|---|---|---|---|---|
| FLUX.2 Klein 4B | 4.6 GB | ~2.6–3 GB | runs (staging alone) | runs | runs great |
| Z-Image Turbo (6B) | 5.9 GB | ~4 GB | needs block streaming (or 3-bit) | runs resident | runs great |
| FLUX.2 Klein 9B | 9.5 GB | ~6 GB | external-SSD stream only | tight | runs great |
| Qwen-Image 2512 | 25.9 GB | ~16 GB | — | — | Mac (24 GB+) |

> The iPhone path is genuine R&D — no one has shipped MLX diffusion this way on a phone.
> Klein 4B 4-bit (two-phase) is the fastest proof point; Z-Image Turbo validates streaming.

### FLUX on iOS — current state & plan

The shipping `flux-2-swift-mlx` is **monolithic** (one `Flux2Pipeline`, no per-block streaming)
and uses macOS-only APIs (`import AppKit` / `NSImage` in its encoders, image processing, and
training), so **today FLUX is the macOS path** and Z-Image (block-streamable) is the iPhone path.

**Decision (2026-06-24): EXTEND `flux-2-swift-mlx` to be cross-platform — do NOT reimplement FLUX
in the Z-Image streaming framework.** A from-scratch reimplementation (a `FluxArchitecture` on the
`MLXDiffusionEngine` streaming seam) is technically attractive — the `mlx-community/flux2-klein-4b-4bit`
checkpoint is mflux-format like Z-Image, so the download / `RangedFileWeightSource` / streaming infra
and the Qwen3-4B encoder all drop straight in — BUT it would create **two FLUX forwards** (Mac vs iOS)
that can drift, and it throws away everything `flux-2-swift-mlx` already supports (dev / klein-4b /
klein-9b, multiple precisions, reference-image conditioning, LoRA, training). One shared implementation
keeps Mac and iOS bit-for-bit consistent and lets future, higher-RAM iPhones run bigger variants.

**What we learned scoping it (2026-06-24):**
- **FLUX.2 Klein 4B components** (from OtterPix + HF): transformer 16-bit 7.2 GB *or* 8-bit 3.3 GB
  pre-quantized; text encoder = **Qwen3-4B** (4-bit 1.9 GB / 8-bit 4.0 GB — the SAME encoder Z-Image
  uses, already iOS-proven); VAE = FLUX-2 small-decoder ~0.58 GB.
- **The 4-bit gotcha:** `flux-2-swift-mlx`'s "4-bit" path *downloads the 7.2 GB bf16 and quantizes
  on load* → loading the bf16 resident would OOM an iPhone. But **`mlx-community/flux2-klein-4b-4bit`
  is a clean PRE-quantized 4-bit checkpoint** (mflux 0.17.5, group size 64, transformer 2.18 GB,
  loads directly, no bf16 spike) — same sharded format the package already loads for 8-bit.
- **Transformer architecture** (verified from the checkpoint header): double+single stream MMDiT —
  keys `x_embedder` / `context_embedder` / `time_guidance_embed` / `transformer_blocks` (double) /
  `single_transformer_blocks` (single) / `double_stream_modulation_{img,txt}` / `single_stream_modulation`.

**Work plan:**
1. **Cross-platform port** (the main effort, moderate — AppKit is concentrated at image boundaries):
   `Package.swift` `[.macOS]` → add `.iOS(.v18)`; replace `NSImage`/`AppKit` with `CGImage` +
   `ImageIO`/`CoreGraphics` in the inference path (output image must port; guard reference-image /
   vision processing + training behind `#if os(macOS)` first — text2img doesn't need them); drop the
   `AppEngines` macOS-only gating so `Flux2DiffusionEngine` is cross-platform.
2. **Add a pre-quantized 4-bit load path** (analogous to the existing pre-quantized 8-bit) so iOS
   loads `mlx-community/flux2-klein-4b-4bit` directly, avoiding the bf16 quantize-on-load spike.
3. **Two-phase resident on iPhone** (the pipeline already has `unloadTextEncoder()`): 4-bit
   transformer 2.18 GB loads directly. **512 fits** (≈ max(encoder 1.9, transformer 2.18 + VAE 0.58)
   + working set ≈ 3.3 GB); **1024 is tight** (double-stream activations push toward ~4.3 GB).
4. **Block streaming (later, Phase 2)** — only needed for 1024 full-res headroom and larger variants
   (Klein 9B, external SSD). Give FLUX the same per-block `WeightSource` ranged-read ladder Z-Image
   uses; peak drops to ~1 resident block + base.

**Interim state (current):** FLUX is excluded from the iOS build via a thin local `AppEngines` wrapper
package that depends on `Flux2DiffusionEngine` **only on macOS**
(`.product(..., condition: .when(platforms: [.macOS]))`). Xcode's `platformFilter` alone does NOT
work — SPM validates the *whole* package graph per platform, so a macOS-only package in the graph
breaks the iOS build; the SPM-native platform-conditional product dependency (expressible only in a
Package manifest, hence the wrapper) is the fix. Removing this gating is step 1 of the port. Z-Image
is today's on-device model.

---

## 3. Architecture

### Package topology (all public)

```
App (this repo — rebuilt)         depends on ↓
  ├── swift-diffusion-core   (NEW public repo: nanguoyu/swift-diffusion-core)
  │     engine protocol · streaming partial-loader · WeightSource · samplers ·
  │     memory governor · catalog + download
  ├── flux-2-swift-mlx       (existing public, MIT — use main branch)
  └── z-image-swift-mlx      (NEW public repo: nanguoyu/z-image-swift-mlx)
```

`swift-diffusion-core` and `z-image-swift-mlx` are standalone public repos (siblings of the
app, like `flux-2-swift-mlx`); the app consumes them as local path dependencies during
development and as versioned git dependencies in release.

### The boundary — `DiffusionEngine`

The app talks only to `DiffusionEngine`. There are **two engine shapes** behind it:

- **`MLXDiffusionEngine`** (in core, iOS + macOS) — drives any *block-streamable*
  `DiffusionArchitecture` and applies the partial-load ladder. This is the path for Z-Image
  and the iPhone.
- **A whole-pipeline facade engine** (macOS-only) — wraps a monolithic pipeline that owns its
  own denoise loop. `flux-2-swift-mlx` is exactly this: it is **macOS-15-only** and exposes one
  `Flux2Pipeline.generateTextToImage(...)` with no per-block access, so it cannot be
  block-streamed. It becomes a `DiffusionEngine` facade in the macOS target — *not* in core
  (core must stay iOS-buildable, and flux-2-swift-mlx can't build for iOS).

`MLXDiffusionEngine` consumes the `DiffusionArchitecture` seam each model package implements:

```
DiffusionEngine        load / generate(progress) / unload / capabilities
   └─ drives ─▶ DiffusionArchitecture   encode() · denoiserBlocks() · decode()
                   each block is a StreamableBlock  (load → run → release)
                   reads weights via WeightSource   (mmap | ranged SSD | hybrid)
```

### `WeightSource` — internal storage *and* external SSD, transparently

Weights are read as byte ranges, not bound to mmap, so the same streaming engine runs from
internal storage or a USB-C external SSD:

- `MmapWeightSource` — mmap the safetensors file (internal, fastest)
- `RangedFileWeightSource` — `pread` byte ranges on demand (external SSD; avoids
  mmap-on-external limits)
- `HybridWeightSource` — hot tensors resident, cold tensors streamed + prefetched

This unlocks running Mac-class models (Klein 9B, even Qwen-Image) off an external SSD on a
phone, at an I/O cost (~1 GB/s over USB 3).

### Memory governor

`DeviceTier` detects chip + `ProcessInfo.physicalMemory` → `(defaultPrecision, cacheLimit,
residentVsStream)`. On iPhone it reads `MemoryProbe.availableBytes()` before building the
pipeline and gates which rung of the ladder is used.

---

## 4. Model catalog

Per-variant layout descriptors (the on-disk layouts differ across families — verified at
least four schemes), so the weight loader dispatches per layout:

| Family | Source (public) | Precisions | License |
|---|---|---|---|
| Z-Image Turbo / Base (6B, S3-DiT, Qwen3-4B encoder) | community MLX repos | 8/4/2-bit | Apache-2.0 |
| FLUX.2 Klein 4B / 9B | mlx-community | 8/4-bit | Apache-2.0 |
| Qwen-Image 2512 | mlx-community | 8/6/5/4/3-bit | Apache-2.0 |

Licenses are encoded per variant and enforced (e.g. FLUX.2 **dev** is non-commercial and is
excluded). Default endpoints: public HuggingFace + user-configurable mirrors. Downloads are
**byte-range resumable** (multi-GB transformers), SHA-verified, with smoothed progress.

---

## 5. UI / UX — dark creative studio

Design language: near-black studio surface, a single violet generative accent, hairline
borders, Tabler outline icons. Two things are first-class because they are what makes this
app special:

1. **Precision is a first-class input** — switching precision live updates size, the
   transformer/encoder/VAE component breakdown, and the hardware-fit badge.
2. **Hardware awareness is everywhere** — a fit badge = `device × model × precision`:
   green *runs great* (resident) · amber *two-phase* / *streams from SSD* · gray *needs more*.

### Four tabs (Mac sidebar / iPhone tab bar)

- **Models** — download center: family-grouped cards, recommended-for-device, precision
  chips, fit badges, install/progress. Model detail drawer: variant table, component
  breakdown, storage location (internal / external SSD), resumable download.
- **Create** — generation workspace: prompt, full-bleed canvas, steps/seed/size, reference
  image (img2img), a memory governor pill (resident / streaming), per-step progress.
- **Library** — your generated images: grid by day, tap for detail (prompt + params),
  **reuse settings** to iterate, favorite, export.
- **Settings** — storage & external SSD: default download location, *stream large models
  from SSD* toggle, per-model location; on iOS the SSD is granted via Files (security-scoped
  bookmark) and must stay connected while generating.

Mac uses `NavigationSplitView`; iOS uses `TabView` + `NavigationStack`; both render the same
shared components.

---

## 6. Open-source / privacy rules

- Use only public dependencies and public, openly-licensed model weights.
- The download/catalog/MLX-abstraction code is adapted from the author's own prior work, but
  everything that lands here is **scrubbed of private identifiers**: no private repo names,
  no private CDN/infra/hostnames, no store/billing IDs, no keys. Endpoints default to public
  HuggingFace + a user-configurable mirror field.

---

## 7. Roadmap

- **Phase 0 — spike & de-risk.** Stand up `swift-diffusion-core` + the `MLXDiffusionEngine`.
  On Mac, run FLUX.2 Klein 4B (via `flux-2-swift-mlx`) and Z-Image Turbo (new package)
  end-to-end. Then on device: measure Klein 4B 4-bit two-phase peak on an 8 GB iPhone and
  Z-Image block-streaming. Prove/disprove MLX-on-iPhone.
  **Status (2026-06-23):** core engine landed and unit-tested — `MLXDiffusionEngine`
  (streaming denoise loop), `FlowMatchEulerSampler`, `SafetensorsWeightSource`,
  `ImageConversion`, `MemoryGovernor`/`DeviceTier`/`MemoryProbe`. Pure-logic tests
  (governor decisions, sampler schedule) pass in CI; MLX-eval tests pass in Xcode (a headless
  CI box has no Metal lib). Remaining: real Z-Image S3-DiT architecture, the macOS FLUX
  facade engine, MLX cache governance, and on-device memory measurement.
- **Phase 1 — Mac app.** Dark-studio shell, Models gallery + detail/download (resumable),
  Create workspace, Library, memory governor, persisted image cache.
- **Phase 2 — iPhone.** Same shell adapted (TabView), `MemoryProbe` gating,
  increased-memory-limit entitlement, internal-storage streaming.
- **Phase 3 — external SSD + breadth.** `WeightSource` ranged-read path (USB-C SSD), more
  model packages (one public repo each), generation queue, downloader hardening.

---

## 8. Migration — what leaves this repo

The original CoreML app is removed once the new path runs (tracked separately; not deleted
until confirmed):

- `MobileDiffuser/ContentView.swift`, `SD3PipelineLoader.swift`, `ModelResourceManager.swift`
  — replaced by the new shell + `swift-diffusion-core`.
- `ml-stable-diffusion/` (vendored CoreML fork) — dropped.
- `scripts/*.py` (CoreML conversion/quantization) — dropped.
- **Kept (ported):** `MemoryProbe` (already in `DiffusionCore`), the partial-load *concept*,
  the increased-memory-limit entitlement, the lazy-build/unload lifecycle.

### Phase 0 open questions

1. Real per-architecture Swift cost for a non-FLUX model (Z-Image S3-DiT + Qwen3-4B encoder).
2. mmap-on-external feasibility on iOS (fall back to `pread` ranged reads).
3. Sustained USB-3 throughput + security-scoped resource lifetime during a long generation.
4. Exact community MLX weight layouts per family (the loader dispatches on layout).
