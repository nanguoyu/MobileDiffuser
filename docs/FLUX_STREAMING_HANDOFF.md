# FLUX.2 1024-on-iPhone block-streaming — status & validation

**The streaming engine AND the app router are fully built, compile green on iOS + macOS, AND the 512
parity gate PASSES bit-identically on Mac** — including the forced per-step streaming path. The only
thing left needs the phone: the **on-device 1024** thermal/memory run.

The streaming path is the resident path's twin — it reuses the same flux-mlx units
(`KleinTextEncoder`, `LatentUtils`, the VAE). Validated:

```
swift run flux2-demo --parity            # resident MLXEngine vs facade  → maxPixelDiff 0, PSNR ∞  ✅
swift run flux2-demo --parity --stream   # per-step block-STREAMING vs facade → maxPixelDiff 0, PSNR ∞  ✅
```

Both are **bit-identical** to the resident facade at 512. The forced-streaming run exercises the exact
load→run→release→clearCache path the iPhone uses at 1024, so the on-device *mechanics* are validated;
only the *physics* (thermal/memory at 1024) remain to test on-device.

---

## What's wired (done, committed, compile-green)

- **Core un-gate** — `MLXDiffusionEngine.capabilities` is memory-driven for FLUX.2 now (no more
  "macOS only"). `swift-diffusion-core@main`.
- **App router** — `AppModel`: `fluxUsesStreaming = isPhone && size > 512`. 512 → resident
  `Flux2FacadeEngine`; 1024 → `MLXDiffusionEngine(architecture: Flux2Architecture(...))` with a
  transformer-only `Flux2ComponentSource.openKlein4BStreaming()`. The loaded streaming flag is in the
  reload key, so crossing 512↔1024 rebuilds the engine. Mac is unaffected (`isPhone` false → facade).
- **The engine** — `Flux2Architecture` / `Flux2Denoiser` / `Flux2StreamableBlock` / `Flux2Weights` /
  `Flux2Sigmas` / `Flux2ComponentSource` in `flux2-diffusion-engine@main`; the streaming decomposition
  + `Flux2StreamingSupport` in `flux-2-swift-mlx@main`.

To build the app, bump the app's two remote pins (`swift-diffusion-core`, `flux-2-swift-mlx`) to
latest `main`; the local-path deps follow automatically. Already done in this branch's (gitignored)
`Package.resolved`. **Never commit a local-path dep into a shared package.**

---

## 1. THE GATE — 512 parity (PASSED on Mac ✅)

Already run and bit-identical (see the two commands above). If you want to reproduce: `cd
flux2-diffusion-engine && swift run flux2-demo --parity [--stream]` (downloads the 4-bit Klein weights
on first run; writes `parity-resident.png` / `parity-streamed.png`). A `--diag` mode compares weights,
the streaming forward, and the encode/init/decode glue tensor-by-tensor.

The one bug found and fixed during validation: this package had pinned an old `swift-diffusion-core`
that predated the architecture-owned sigma hook, so the engine silently fell back to the fixed-shift
sampler schedule (step-3 σ ~0.001 vs FLUX's ~0.717) and produced a coherent-but-different image. The
pin is bumped; keep shared-package pins current.

---

## 2. Then on-device 1024 (iPhone) — the only step left

Once 512 parity is green, build the app to your iPhone and render at 1024 — the router selects the
streaming engine automatically. Instrument: `thermalState` transitions, per-step wall-clock,
`MemoryProbe` peak. Expect slower + hotter than 512 (intrinsic 4× FLOPs + ~8.7 GB pread/image); the
win is it **completes or auto-pauses to cool** (Wave 1's ThermalGovernor) instead of restarting.
Budget for ~a few on-device fixes (the Z-Image streaming bring-up needed 4).

Tunables if tight: `Flux2StreamableBlock.approximateBytes` (residency planning), the streaming
`cacheLimit` (384 MB in `MLXDiffusionEngine.load`), the VAE tile overlap (bump `.aggressive` 4 → 8 if
seams show).

---

## What's proven offline (no checkpoint) vs to validate

| Piece | Status | Test |
|---|---|---|
| Block-streaming decomposition == monolithic forward | ✅ proven 1e-4 | `Flux2Core/StreamDecompositionTests` |
| FLUX sigma schedule (exact values) | ✅ proven | `Flux2DiffusionEngine/Flux2SigmasTests` |
| Per-block + shared disk↔module key bijection | ✅ proven | `Flux2DiffusionEngine/Flux2WeightsTests` |
| Denoiser wiring (holder/adapter) == monolithic | ✅ proven 1e-4 | `Flux2DiffusionEngine/Flux2DenoiserTests` |
| Component-source routing | ✅ proven | `Flux2DiffusionEngine/Flux2ComponentSourceTests` |
| App router + un-gate | ✅ compile-green iOS + macOS | (build) |
| encode / initialLatent / decode end-to-end parity | ✅ **bit-identical on Mac** | `flux2-demo --parity` |
| per-step streaming load/release parity | ✅ **bit-identical on Mac** | `flux2-demo --parity --stream` |
| 1024 thermal/memory survival | ⏳ on-device | §2 |

`reuse-shell` (the ~100-quantize-passes optimization) stays deferred per the audit until 512 parity is
green and a dedicated bit-exact parity test covers it.
