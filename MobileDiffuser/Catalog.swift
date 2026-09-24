// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import DiffusionCore

/// The built-in model catalog. Z-Image runs everywhere (downloaded in-app via `ModelDownloader`);
/// FLUX.2 Klein 4B also runs everywhere now — the `flux-2-swift-mlx` pipeline self-downloads and
/// self-loads its weights (Mac quantizes bf16 on load; iPhone loads the pre-quantized 4-bit
/// checkpoint via the two-phase pipeline). Both entries ship on both platforms.
enum Catalog {
    static let zImageTurbo = DiffusionModel(
        id: "z-image-turbo-q4",
        displayName: "Z-Image Turbo (6B)",
        family: .zImage,
        publisher: "Tongyi (Alibaba)",
        summary: "4-bit · 8-step · S3-DiT + Qwen3-4B",
        license: .apache2,
        architecture: ArchitectureSpec(family: .zImage, latentChannels: 16,
            defaultSampler: .flowMatchEuler, defaultSteps: 8, defaultGuidance: 1.0,
            vaeScale: 0.3611, vaeShift: 0.1159),
        variants: [ModelVariant(precision: .q4, approximateBytes: 5_900_000_000,
            components: ComponentSizes(transformer: 3_460_000_000, textEncoder: 2_260_000_000, vae: 160_000_000),
            layout: .mfluxShard,
            source: ModelSource(huggingFaceRepo: "deepsweet/Z-Image-Turbo-6B-MLX-Q4"))])

    static let flux2Klein = DiffusionModel(
        id: "flux2-klein-4b",
        displayName: "FLUX.2 Klein (4B)",
        family: .flux2,
        publisher: "Black Forest Labs",
        summary: "Selectable precision · Qwen3-4B encoder",
        license: .apache2,
        // FLUX.2 Klein 4B is step-distilled: native 4 steps, guidance 1.0 (verified vs the HF model
        // card `num_inference_steps=4` and flux-2-swift-mlx `Flux2Config.klein4B.defaultSteps == 4`).
        architecture: ArchitectureSpec(family: .flux2, latentChannels: 16,
            defaultSampler: .flowMatchEuler, defaultSteps: 4, defaultGuidance: 1.0),
        // The facade resolves the real transformer repo per platform — Mac quantizes the
        // black-forest-labs bf16 file on load; iPhone loads mlx-community/flux2-klein-4b-4bit
        // (pre-quantized, no spike). `source`/`layout` here are informational: the facade self-manages
        // its weights and ignores them. `components` feed the fit-badge memory estimate.
        variants: [ModelVariant(precision: .q4, approximateBytes: 4_600_000_000,
            components: ComponentSizes(transformer: 2_180_000_000, textEncoder: 2_260_000_000, vae: 170_000_000),
            layout: .mfluxShard,
            source: ModelSource(huggingFaceRepo: "mlx-community/flux2-klein-4b-4bit"))])

    /// Qwen-Image 2.1 runs on the stable-diffusion.cpp engine from GGUF files: the denoiser and the
    /// Qwen3-VL text encoder come in several quantizations (picked in the model details), the VAE is
    /// a single bf16 file. Sizes below are the defaults, Q4_K_M + UD-Q4_K_XL.
    static let qwenImage21 = DiffusionModel(
        id: "qwen-image-2.1-gguf",
        displayName: "Qwen-Image 2.1 (7B)",
        family: .qwenImage,
        publisher: "Qwen (Alibaba)",
        summary: "GGUF on stable-diffusion.cpp, Qwen3-VL-8B encoder",
        license: .other(name: "Qwen Research", commercialUse: false),
        // Sampled without classifier-free guidance, as the reference pipeline does (diffusers'
        // QwenImage21Pipeline defaults true_cfg_scale to 1.0), so each step is one denoiser pass.
        // Guidance 6 halves the speed and, at 10 steps, turns the image blotchy and speckled.
        architecture: ArchitectureSpec(family: .qwenImage, latentChannels: 64,
            defaultSampler: .flowMatchEuler, defaultSteps: 20, defaultGuidance: 1.0),
        variants: [ModelVariant(precision: .q4, approximateBytes: 10_023_774_200,
            components: ComponentSizes(transformer: 4_199_565_024, textEncoder: 5_148_699_488, vae: 675_509_688),
            layout: .flatSingle,
            source: ModelSource(huggingFaceRepo: QwenImage21Files.transformerRepo))])

    static var all: [DiffusionModel] { [zImageTurbo, flux2Klein, qwenImage21] }
}

extension ModelFamily {
    /// The short family name on model cards.
    var label: String {
        switch self {
        case .zImage: "Z-Image"
        case .flux2: "FLUX.2"
        case .qwenImage: "Qwen-Image"
        }
    }
}

/// Sampling steps for one model on one kind of device: the default and the options in Create.
struct StepSettings {
    let initial: Int
    let choices: [Int]
}

extension Catalog {
    /// Steps per model on a Mac and on a phone. The distilled models are step-sensitive and run
    /// around their native count everywhere. Qwen-Image 2.1 is not distilled: a Mac starts at the 20
    /// steps it is validated with (40 is the model card's full-quality setting); a phone takes about
    /// a minute per step, so it starts at 10 and stops at 20.
    static let steps: [String: (mac: StepSettings, phone: StepSettings)] = [
        zImageTurbo.id: (mac: StepSettings(initial: 8, choices: [4, 8, 16]),
                         phone: StepSettings(initial: 8, choices: [4, 8, 16])),
        flux2Klein.id: (mac: StepSettings(initial: 4, choices: [2, 4, 8]),
                        phone: StepSettings(initial: 4, choices: [2, 4, 8])),
        qwenImage21.id: (mac: StepSettings(initial: 20, choices: [10, 20, 40]),
                         phone: StepSettings(initial: 10, choices: [10, 15, 20])),
    ]
}

extension DiffusionModel {
    /// This model's steps on this kind of device. A model missing from the table gets its native
    /// count with half and double as the other options.
    func stepSettings(onPhone: Bool) -> StepSettings {
        if let steps = Catalog.steps[id] { return onPhone ? steps.phone : steps.mac }
        let n = max(2, architecture.defaultSteps)
        return StepSettings(initial: n, choices: Array(Set([max(2, n / 2), n, n * 2])).sorted())
    }

    /// Render-size options (px). Current models handle this square range; native is the top.
    var sizeChoices: [Int] { [512, 768, 1024] }
    var nativeSize: Int { 1024 }
}
