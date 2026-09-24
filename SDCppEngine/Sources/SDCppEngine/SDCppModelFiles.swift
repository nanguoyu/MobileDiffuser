// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import Foundation

/// The on-disk pieces an sd.cpp model is assembled from. Unlike the MLX engines, which read one
/// repository through a `WeightSource`, sd.cpp takes separate files that usually come from different
/// repositories (a GGUF denoiser, a GGUF text encoder, a safetensors VAE), so the app hands them over
/// as explicit paths.
public struct SDCppModelFiles: Sendable, Hashable {
    /// The denoiser (DiT or UNet), GGUF or safetensors.
    public var diffusionModel: URL
    /// The language-model text encoder (for Qwen-Image, Qwen3-VL), GGUF or safetensors.
    public var textEncoder: URL
    public var vae: URL
    /// The text encoder's vision tower (`mmproj`). Only reference-image editing needs it, so a
    /// text-to-image install can leave it out.
    public var textEncoderVision: URL?

    public init(diffusionModel: URL, textEncoder: URL, vae: URL, textEncoderVision: URL? = nil) {
        self.diffusionModel = diffusionModel
        self.textEncoder = textEncoder
        self.vae = vae
        self.textEncoderVision = textEncoderVision
    }

    var required: [URL] { [diffusionModel, textEncoder, vae] }
}

/// Runtime knobs passed to sd.cpp when the context is created.
public struct SDCppOptions: Sendable, Hashable {
    /// Map weight files instead of reading them into anonymous memory. Mapped pages are clean and
    /// file-backed, so iOS can reclaim them under pressure instead of terminating the app.
    public var memoryMapWeights: Bool
    public var flashAttention: Bool
    /// Upper bound, in GiB, for the weights and buffers sd.cpp keeps on the GPU. When a component
    /// does not fit, sd.cpp runs it in segments and prefetches the next segment's weights. `nil` uses
    /// the engine's default for the device: 30% of the RAM on a phone, and on a Mac whatever sd.cpp
    /// sizes itself from the memory that is free when the model loads.
    public var gpuBudgetGiB: Double?
    /// Decode the latent in tiles: slower, but the decoder's workspace stays that of one tile. `nil`
    /// tiles only the renders whose untiled decode would take a large share of the device's memory.
    public var tiledVAEDecode: Bool?

    public init(memoryMapWeights: Bool = true, flashAttention: Bool = true,
                gpuBudgetGiB: Double? = nil, tiledVAEDecode: Bool? = nil) {
        self.memoryMapWeights = memoryMapWeights
        self.flashAttention = flashAttention
        self.gpuBudgetGiB = gpuBudgetGiB
        self.tiledVAEDecode = tiledVAEDecode
    }
}

public enum SDCppError: LocalizedError, Equatable {
    case missingFile(String)
    case loadFailed(String)
    case notLoaded
    case generationFailed(String)
    case editingNeedsVisionEncoder
    case unreadableImage

    public var errorDescription: String? {
        switch self {
        case .missingFile(let name):
            return "A model file is missing: \(name). Download the model again."
        case .loadFailed(let detail):
            return detail.isEmpty ? "The model could not be loaded." : "The model could not be loaded: \(detail)"
        case .notLoaded:
            return "The model is not loaded."
        case .generationFailed(let detail):
            return detail.isEmpty ? "The image could not be generated." : "The image could not be generated: \(detail)"
        case .editingNeedsVisionEncoder:
            return "Editing with a reference image needs the vision encoder. Download it in the model details."
        case .unreadableImage:
            return "The reference image could not be read."
        }
    }
}
