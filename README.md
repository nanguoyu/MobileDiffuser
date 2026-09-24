<h1>
  <img src="MobileDiffuser/Assets.xcassets/AppIcon.appiconset/AppIcon.png" alt="MobileDiffuser app icon" width="64" align="left">
  MobileDiffuser
</h1>

<br>

MobileDiffuser is a **universal (macOS + iOS) on-device image-generation app built in Swift on
MLX**, with [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) as a second
engine for GGUF models — no Core ML, no Python at runtime. It runs open-weight diffusion models
locally, with first-class in-app model management/download and per-hardware memory management.

> MobileDiffuser began as Wenwu Tang's on-device **Core ML Stable Diffusion 3** app; this is its
> ground-up rebuild onto a pure-**MLX** engine. The Core ML / `ml-stable-diffusion` stack is gone, and
> MLX now runs everywhere, including the iPhone partial-load path (see [Attribution](#attribution)).

## Status

- **macOS: working.** The app builds and generates behind one studio UI, for both text-to-image
  and image-to-image.
  - **Z-Image Turbo (6B)** — single-stream S3-DiT + Qwen3-4B text encoder, 8-step, 4-bit. Pure
    Swift+MLX port; validated end-to-end on macOS and on iPhone through block streaming.
  - **FLUX.2 Klein (4B)** — cross-platform facade over `flux-2-swift-mlx`; 4-bit uses the
    pre-quantized `mlx-community/flux2-klein-4b-4bit` checkpoint on both Mac and iPhone.
  - **Qwen-Image 2.1 (7B)** — GGUF on the stable-diffusion.cpp engine, Qwen3-VL-8B text encoder,
    20 steps without guidance, as the reference pipeline samples it. On an M1 Pro (Q4_K_M +
    UD-Q4_K_XL) a 512px render takes about 3 minutes and a 1024px one about 14; memory-mapped
    weights keep the process footprint near 2 GB.
  - **Image-to-image** is FLUX.2 reference-context: 1–3 reference images are VAE-encoded and
    concatenated into the transformer sequence as conditioning, and the output denoises from pure
    noise while attending to them (editing / style / composition — not a strength slider). On Mac,
    references run up to the chosen size through the resident facade.
- **iOS: working on an iPhone 16 Pro (8 GB), validated end-to-end.**
  - Z-Image Turbo 6B runs on iPhone via block-streaming partial load (about 2.2 GB peak).
  - FLUX.2 Klein 4B text-to-image:
    - **512px** — resident facade, about 4.3 GB peak, about 1m11s.
    - **1024px** — block-streaming transformer (one block resident at a time) plus seam-free,
      bit-exact conv-striped VAE decode, about 3.83 GB peak, about 4m22s. A cheap latent preview
      shows the image forming during generation.
  - Qwen-Image 2.1 7B, 512px: 10 steps (the phone's default) in about 5.5 minutes, peak about
    2.6 GB. On a phone sd.cpp runs under a GPU budget of 30% of the RAM, so the text encoder and the
    denoiser run in segments that read their weights from flash as they go.
  - FLUX.2 Klein 4B **image-to-image** (reference-context, 512px) runs on iPhone via the
    block-streaming path — about 3.45 GB peak, about 1m49s, validated on-device. The streamed
    output is pixel-identical to the resident Mac facade (parity-gated).
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
- [`SDCppEngine`](SDCppEngine) (in this repo) — the same `DiffusionEngine` boundary over
  [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) on its Metal backend. It runs
  GGUF models, which MLX cannot read, and brings sd.cpp's model implementations with it, so a model
  sd.cpp supports needs no port. Qwen-Image 2.1 is the first. Weights are memory-mapped straight into
  Metal buffers, so they stay reclaimable file pages instead of process memory. The sd.cpp fix that
  makes this work on Metal is applied from [`scripts/sdcpp-patches`](scripts/sdcpp-patches) until
  upstream merges it ([leejet/stable-diffusion.cpp#2037](https://github.com/leejet/stable-diffusion.cpp/pull/2037)).

Both models run through a `DiffusionEngine` facade, so the studio switches between them uniformly.
Z-Image uses the generic block-streaming `MLXDiffusionEngine` on iPhone and a resident facade on
Mac. FLUX.2 runs through its cross-platform facade on both platforms; on Mac it stays resident,
while on iPhone the heavier paths (1024px text-to-image and image-to-image) stream the transformer
one block at a time. For image-to-image, the streamed sequence carries the reference tokens
alongside the output, only the output tokens are denoised and decoded, and the reference VAE is
freed before the transformer streams — keeping the phone under its memory budget.

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
   the local packages and their public remote dependencies; the prebuilt stable-diffusion.cpp
   XCFramework comes from this repository's releases.
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

To change the stable-diffusion.cpp version or its patches, run `./scripts/build-sdcpp-xcframework.sh`
(needs CMake, `brew install cmake`). The framework it builds into `SDCppEngine/Vendor` takes
precedence over the released one; publish its zip as a new release and point
`SDCppEngine/Package.swift` at it with the checksum the script prints.

## License

MobileDiffuser is licensed under the **Mozilla Public License 2.0** (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE)), retaining the original Core ML release's MIT copyright. Model weights are subject to
their own licenses (Z-Image Turbo and FLUX.2 Klein are Apache-2.0; Qwen-Image 2.1 is under the Qwen
Research License, non-commercial) and are not included in this repository. stable-diffusion.cpp and
ggml are MIT-licensed; the build script fetches them, so they are not included either.

## Attribution

The original on-device **Core ML Stable Diffusion 3** MobileDiffuser is **Wenwu Tang's**
([TWWinde/MobileDiffuser](https://github.com/TWWinde/MobileDiffuser), with Olga Saukh). This
**pure-MLX rebuild** is **Dong Wang's**. Citation for the original Core ML work:

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
