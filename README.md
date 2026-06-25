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
> new engine is MLX everywhere, including the iPhone partial-load path.

## Status

- **macOS: working.** The app builds and generates with two models behind one studio UI.
  - **Z-Image Turbo (6B)** — single-stream S3-DiT + Qwen3-4B text encoder, 8-step, 4-bit. Pure
    Swift+MLX port; validated end-to-end on macOS and on iPhone through block streaming.
  - **FLUX.2 Klein (4B)** — cross-platform facade over `flux-2-swift-mlx`; 4-bit uses the
    pre-quantized `mlx-community/flux2-klein-4b-4bit` checkpoint on both Mac and iPhone.
- **iOS: working for the current 512px paths.**
  - Z-Image Turbo 6B runs on iPhone via block-streaming partial load.
  - FLUX.2 Klein 4B was validated on an iPhone 16 Pro at 512px / 4 steps / small decoder
    (about 4.3 GB peak resident memory, about 1m11s). 1024px remains cautious because activations
    scale much higher.
- Models are **downloaded inside the app** (from Hugging Face) into Application Support — no
  Python, pip, Git LFS, or CLI required.

## Architecture

The app talks to a single boundary, `DiffusionEngine`, and never imports a specific model:

- [`swift-diffusion-core`](https://github.com/nanguoyu/swift-diffusion-core) — the engine
  protocol, the streaming/partial-load ladder, `WeightSource`, samplers, and the memory governor.
- [`z-image-swift-mlx`](https://github.com/nanguoyu/z-image-swift-mlx) — Z-Image (S3-DiT +
  Qwen3-4B + AutoencoderKL) reimplemented in MLX, with `ZImagePipeline` / `ZImageFacadeEngine`
  and in-app model download.
- [`flux2-diffusion-engine`](https://github.com/nanguoyu/flux2-diffusion-engine) — a cross-platform facade
  over [`flux-2-swift-mlx`](https://github.com/nanguoyu/flux-2-swift-mlx) for FLUX.2.

Both models run through a `DiffusionEngine` facade, so the studio switches between them uniformly.
Z-Image uses the generic block-streaming `MLXDiffusionEngine` on iPhone and a resident facade on
Mac. FLUX.2 uses its whole-pipeline facade on both platforms; iPhone runs the phone-aware two-phase
4-bit path.

## Requirements

- **macOS 14+**, Xcode 16.2+ (Apple Silicon).
- **iOS 18.2 app target** in the Xcode project. The package layer is iOS 17+, but the app target
  currently builds against 18.2.
- An Apple Developer account to run on a device (signing below).
- Current development checkouts expect sibling repositories next to this one:
  `../z-image-swift-mlx` and `../flux2-diffusion-engine`.

## Build & run

1. Clone this repo plus the sibling engine repos:
   ```bash
   git clone https://github.com/nanguoyu/MobileDiffuser
   git clone https://github.com/nanguoyu/z-image-swift-mlx
   git clone https://github.com/nanguoyu/flux2-diffusion-engine
   ```
   They should sit under the same parent directory because the Xcode project uses local package
   references during development.
2. Open `MobileDiffuser.xcodeproj` in Xcode. Swift Package dependencies resolve automatically from
   the local packages and their public remote dependencies.
3. **Signing — keeps your Apple Team ID out of the repo:**
   ```bash
   cp Signing.xcconfig.example Signing.xcconfig     # Signing.xcconfig is gitignored
   # edit Signing.xcconfig: set DEVELOPMENT_TEAM = <your team id>
   ```
   The project reads code-signing from this gitignored file, so your identity never lands in the
   tracked project file.
4. Build & run on macOS or a physical iPhone.
5. In the app: open **Models**, pick a recipe, download the missing components, enter a prompt,
   and **Generate**.

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
