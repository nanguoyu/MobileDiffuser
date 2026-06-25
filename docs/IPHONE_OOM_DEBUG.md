# iPhone Memory and MLX Debugging Notes

This document summarizes practical debugging for the current Swift + MLX iPhone
paths. The old Core ML / ANE split-stage notes no longer describe the running
app.

## Current Assumptions

- Backend: MLX for Swift.
- Models: Z-Image Turbo 6B and FLUX.2 Klein 4B.
- Default iPhone render size: 512px.
- Z-Image iPhone residency: block streaming through `MLXDiffusionEngine`.
- FLUX.2 iPhone residency: two-phase whole-pipeline facade.
- App entitlement: `com.apple.developer.kernel.increased-memory-limit`.
- Memory readout: `MemoryProbe.residentBytes()` sampled during generation.

## Validated iPhone Paths

```text
Z-Image Turbo 6B:
  iPhone 16 Pro
  block streaming
  about 2.2 GB peak resident memory in the validated run

FLUX.2 Klein 4B:
  iPhone 16 Pro
  4-bit pre-quantized transformer
  4-bit Qwen3 encoder
  small decoder
  512px, 4 steps
  about 4.3 GB peak resident memory
  about 1m11s
```

Treat 1024px iPhone generation as experimental until measured. Activations and
VAE decode scale much faster than the weight sizes suggest.

## Why OOM Happens

Common causes:

1. The app accidentally runs a resident plan instead of streaming.
2. A `WeightSource` does not free on release, so block streaming does not
   actually lower peak memory.
3. Text encoder weights remain referenced after encode.
4. A FLUX recipe uses a large decoder or high resolution while the transformer
   remains resident.
5. MLX GPU cache is not trimmed often enough for a short denoise loop.
6. A partial or empty download is treated as installed, then fails during load.
7. Device thermal pressure lowers usable headroom during a long run.

## What To Watch In The App

The Create model bar shows:

- phase: downloading, loading, generating, done, failed,
- step progress,
- generation duration after success,
- peak resident memory after a run.

The peak readout is the most useful number:

```text
peak X.Y / Z.Y GB
```

`X.Y` is measured `phys_footprint`. `Z.Y` is the app's conservative budget from
`DeviceTier`, not an exact jetsam limit.

## Jetsam Logs

On iPhone:

1. Open Settings.
2. Go to Privacy & Security.
3. Open Analytics & Improvements.
4. Open Analytics Data.
5. Search for `MobileDiffuser` or `JetsamEvent`.
6. Share the `.ips` file.

In Xcode:

1. Connect the iPhone.
2. Open Window -> Devices and Simulators.
3. Select the device.
4. Open View Device Logs.
5. Search for `MobileDiffuser` or `Jetsam`.

Important fields:

```text
reason
phys_footprint
rpages
largestProcess
frontmost
high_water_mark
thermal state
```

`phys_footprint` is the number to compare against the app's peak resident memory
readout.

## Command-Line Device Debugging

When Xcode console output is not enough, use `devicectl` with a compatible
Xcode for the device OS:

```bash
xcrun devicectl device list
xcrun devicectl device process launch --device <device-id> --console <bundle-id>
```

To inspect app data or logs:

```bash
xcrun devicectl device copy from \
  --device <device-id> \
  --domain-type appDataContainer \
  --domain-identifier <bundle-id> \
  --source <path-inside-container> \
  --destination <local-path>
```

Use an Xcode version that supports the iOS version on the device. If the device
appears offline only because it runs a newer iOS beta, install the matching
Xcode beta and select it with `DEVELOPER_DIR` when needed.

## Z-Image Streaming Checks

If Z-Image OOMs or fails on iPhone, verify:

1. The fit badge says `Streams`, not `Runs great` or `Two-phase`.
2. `AppModel.zImageUsesStreaming` is true on iPhone.
3. `ZImageComponentSource.open(..., streaming: true)` is used.
4. The transformer sub-source is `RangedFileWeightSource`.
5. The source reports `freesOnRelease == true`.
6. `ZImageComponentSource.tensor()` routes bare `layers.*` keys to the
   transformer sub-source.
7. `encode` calls `releaseComponent(.textEncoder)` after materializing the
   prompt embeddings.
8. `releaseTextEncoder()` drops the Qwen3 module and clears the MLX cache.

If the failure is a missing `layers.N.*` tensor, suspect component-source
routing or a partial/corrupt Z-Image download.

## FLUX.2 Checks

For iPhone, the expected safe recipe is:

```text
transformer: 4-bit
text encoder: 4-bit
decoder: small decoder
size: 512
steps: 4
```

Verify:

1. 4-bit transformer resolves to `mlx-community/flux2-klein-4b-4bit`.
2. The pipeline loads the pre-quantized checkpoint directly.
3. The text encoder unloads before denoise.
4. iPhone clears MLX GPU cache every denoise step.
5. Standard VAE is only used at 512px on iPhone.
6. Switching decoder reloads the loaded FLUX recipe.

If a run fails during final decode, suspect VAE activation peak. The standard
VAE is sharper but has riskier activations than the small decoder.

## Download Failure Checks

For Z-Image:

- the app clears `URLCache.shared` on iOS before Hugging Face snapshot work,
- an empty file listing is an error,
- install state verifies all shards referenced by each component index,
- `*.incomplete` markers make the model not installed.

Symptoms of a bad download:

```text
download completes too quickly
model folder has only JSON files
missing tensor during load
component folder has no safetensors
```

Tap download again to resume or remove the model and re-download.

## Reporting Format

Use this format for memory or generation reports:

```text
device:
iOS version:
Xcode version:
model:
recipe:
size:
steps:
seed:
time:
peak resident readout:
last visible app status:
last console log:
jetsam phys_footprint:
thermal state:
```

Reports are only comparable when size, steps, decoder, transformer precision,
and encoder precision are included.
