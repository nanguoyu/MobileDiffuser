# Technical Report

MobileDiffuser is now a pure Swift + MLX experiment in running open-weight
diffusion models locally on Apple devices. The current goal is one universal app
for macOS and iPhone, with model management and memory-aware loading as first
class product features.

The previous Core ML / SD3 implementation has been removed from the app. It
remains relevant only as historical attribution and as background for the
partial-load idea.

For implementation details, read:

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [REPRODUCING_MODELS.md](REPRODUCING_MODELS.md)
- [IPHONE_OOM_DEBUG.md](IPHONE_OOM_DEBUG.md)
- [BLUEPRINT.md](BLUEPRINT.md)

## Goals

1. Run image generation locally on Mac and iPhone.
2. Use one Swift + MLX model boundary rather than separate Core ML and MLX
   stacks.
3. Keep large models inside iPhone memory limits through two-phase loading and
   block streaming.
4. Make model download, precision, storage, and hardware fit visible in the UI.
5. Use public dependencies and open-weight model sources.

## Current Models

### Z-Image Turbo 6B

Z-Image Turbo is a 6B S3-DiT model with a Qwen3-4B text encoder and a
FLUX-family VAE. The app uses the 4-bit MLX checkpoint from
`deepsweet/Z-Image-Turbo-6B-MLX-Q4`.

Status:

- macOS resident path generates coherent 1024px images.
- iPhone block-streaming path has been validated on iPhone 16 Pro.
- Measured peak for the iPhone streaming path was about 2.2 GB in the validated
  run.

Main technical finding: block streaming only saves memory when the source
actually frees buffers on release. The app therefore uses a `RangedFileWeightSource`
for streamed transformer blocks and refuses streaming residency with a resident
source.

### FLUX.2 Klein 4B

FLUX.2 Klein runs through a whole-pipeline facade over `flux-2-swift-mlx`.
It is not block-streamed by the generic engine because the upstream pipeline owns
its denoise loop.

Status:

- macOS path works with the pre-quantized 4-bit checkpoint.
- iPhone path works at 512px / 4 steps / small decoder.
- Validated iPhone 16 Pro run: about 4.3 GB peak resident memory, about 1m11s,
  clean image.

Main technical finding: the normal "4-bit" load path must not download bf16 and
quantize in memory on iPhone. The app uses the pre-quantized
`mlx-community/flux2-klein-4b-4bit` checkpoint and loads packed 4-bit tensors
directly.

## Main Findings

### One engine boundary keeps the app simple

The UI only knows about `DiffusionEngine`, `GenerationRequest`, progress, and
capabilities. Z-Image and FLUX are very different internally, but the app can
download, load, switch, unload, and generate through one state machine.

### Two-phase loading is necessary but not sufficient

Unloading the text encoder before transformer/VAE generation prevents the
Qwen3 encoder and transformer from co-residing. That is enough for FLUX.2 Klein
4B at 512px on the tested iPhone.

For Z-Image Turbo 6B, the transformer is too large for the 8 GB phone budget in
a resident plan. It needs block streaming so only one transformer block is
resident at a time.

### Streaming requires exact ownership boundaries

Z-Image has three component trees with colliding tensor names:

```text
text_encoder/
transformer/
vae/
```

The generic engine accepts one `WeightSource`, so Z-Image uses a composite
`ZImageComponentSource`. Bare keys are routed to the transformer so the generic
streaming loop can call `block.load(from: source)` without model-specific
knowledge.

This was a load-bearing bug fix: requiring a component prefix for every tensor
made streamed blocks fail to find `layers.*` weights.

### Text encoder source lifetime matters

Dropping the text encoder module alone did not free all memory when the
composite source still held the text-encoder safetensors arrays. Z-Image now
drops the text-encoder sub-source after `encode`, then `releaseTextEncoder`
drops the module. Both references must be gone before the memory is reclaimed.

### Quality bugs are not always quantization bugs

Z-Image's grainy/mosaic output was traced to VAE GroupNorm channel grouping.
The fix was PyTorch-compatible grouping in the VAE, plus mflux-parity fixes for
the sampler, timestep conventions, Qwen3 precision, caption padding, and AdaLN.

The 4-bit checkpoint itself was not the root cause.

### Memory estimates need empirical calibration

The iPhone `DeviceTier` budget is deliberately conservative. FLUX.2 Klein
measured about 4.3 GB peak on an 8 GB iPhone and did not jetsam, even though the
displayed budget was lower. Fit badges should be treated as conservative
guidance, not a hard OS limit.

## Runtime Defaults

```text
Z-Image Turbo:
  precision: 4-bit
  native steps: 8
  macOS default size: 1024
  iPhone default size: 512
  iPhone residency: block streaming

FLUX.2 Klein:
  iPhone transformer: 4-bit pre-quantized
  iPhone text encoder: 4-bit
  macOS default transformer: 8-bit
  default decoder: small decoder
  native steps: 4
  guidance: 1.0
```

## Validation Notes

Use Xcode or `xcodebuild` for MLX validation where possible. Plain `swift test`
can fail because CLI builds may not have MLX's `default.metallib` next to the
test executable.

Useful report format:

```text
device:
OS version:
Xcode version:
model:
recipe:
size:
steps:
seed:
generation time:
peak resident memory:
last app status:
jetsam log, if any:
```

## Current Limitations

- The in-app Library is session-only.
- img2img is not exposed in the Create UI.
- External SSD model streaming is not exposed in the app yet.
- FLUX.2 1024px on iPhone is still cautious because VAE decode and attention
  activations scale sharply.
- Capability estimates are conservative and should keep being updated from
  real device measurements.
