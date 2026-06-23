<h1>
  <img src="MobileDiffuser/Assets.xcassets/AppIcon.appiconset/AppIcon.png" alt="MobileDiffuser app icon" width="64" align="left">
  MobileDiffuser
</h1>

<br>

MobileDiffuser is a **universal (macOS + iOS) on-device image-generation app built in pure
Swift + MLX** — no Core ML, no Python at runtime. It runs open-weight diffusion models locally,
with first-class in-app model management/download and per-hardware memory management.

> This is a ground-up rebuild of the original Core ML Stable Diffusion 3 app (see
> [Attribution](#attribution)). The Core ML / `ml-stable-diffusion` stack has been removed; the
> new engine is MLX everywhere, including the planned iPhone partial-load path.

## Status

- **macOS: working.** The app builds and generates with two models behind one studio UI.
  - **Z-Image Turbo (6B)** — single-stream S3-DiT + Qwen3-4B text encoder, 8-step, 4-bit. Pure
    Swift+MLX port; validated end-to-end (coherent 1024×1024 images). Runs on macOS; iOS is the
    target.
  - **FLUX.2 Klein (4B)** — macOS-only (its pipeline is monolithic and macOS-native).
- **iOS: in progress.** The app is universal and Z-Image is designed to run on iPhone via
  block-streaming partial load; the on-device path is still being brought up.
- Models are **downloaded inside the app** (from Hugging Face) into Application Support — no
  Python, pip, Git LFS, or CLI required.

## Architecture

The app talks to a single boundary, `DiffusionEngine`, and never imports a specific model:

- [`swift-diffusion-core`](https://github.com/nanguoyu/swift-diffusion-core) — the engine
  protocol, the streaming/partial-load ladder, `WeightSource`, samplers, and the memory governor.
- [`z-image-swift-mlx`](https://github.com/nanguoyu/z-image-swift-mlx) — Z-Image (S3-DiT +
  Qwen3-4B + AutoencoderKL) reimplemented in MLX, with `ZImagePipeline` / `ZImageFacadeEngine`
  and in-app model download.
- [`flux2-diffusion-engine`](https://github.com/nanguoyu/flux2-diffusion-engine) — a macOS facade
  over [`flux-2-swift-mlx`](https://github.com/nanguoyu/flux-2-swift-mlx) for FLUX.2.

Both models run through a `DiffusionEngine` facade, so the studio switches between them uniformly.
FLUX is compiled in only on macOS (`#if os(macOS)` + a macOS platform filter on the package), so
the iOS build excludes it.

## Requirements

- **macOS 14+**, Xcode 16.2+ (Apple Silicon). FLUX.2 is macOS-only; Z-Image targets macOS + iOS.
- **iOS 17+** deployment for the iPhone path (in progress).
- An Apple Developer account to run on a device (signing below).

## Build & run

1. Clone and open `MobileDiffuser.xcodeproj` in Xcode (Swift Package dependencies resolve
   automatically).
2. **Signing — keeps your Apple Team ID out of the repo:**
   ```bash
   cp Signing.xcconfig.example Signing.xcconfig     # Signing.xcconfig is gitignored
   # edit Signing.xcconfig: set DEVELOPMENT_TEAM = <your team id>
   ```
   The project reads code-signing from this gitignored file, so your identity never lands in the
   tracked project file.
3. Build & run (macOS, or an iOS device for the Z-Image path).
4. In the app: pick a model and tap **Download** (FLUX self-downloads on first generate), enter a
   prompt, and **Generate**.

## License

This fork is licensed under the **Mozilla Public License 2.0** (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE)), retaining the original project's MIT copyright. Model weights are subject to
their own licenses (Z-Image Turbo and FLUX.2 Klein are Apache-2.0) and are not included in this
repository.

## Attribution

Forked from [TWWinde/MobileDiffuser](https://github.com/TWWinde/MobileDiffuser), an on-device
Core ML Stable Diffusion 3 app. Citation for the original Core ML work:

```bibtex
@misc{tang2025mobilediffuser,
  author       = {Wenwu Tang and Dong Wang and Olga Saukh},
  title        = {MobileDiffuser: On-device Stable Diffusion 3 Medium on iPhone with Core ML},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/TWWinde/MobileDiffuser}}
}
```
