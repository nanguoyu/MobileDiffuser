// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import DiffusionCore
import Foundation

/// How much memory one render needs, phase by phase.
///
/// Encoding holds the text encoder, denoising the denoiser plus its compute buffer, decoding the VAE
/// plus its workspace; the peak is the largest phase, because the weights are memory-mapped and the
/// page cache can take a finished phase's weights back. Under a GPU budget (phones, see
/// `SDCppDiffusionEngine.defaultGPUBudgetGiB(on:)`) sd.cpp keeps at most that much of a component
/// resident and runs it in segments, reading the rest of its weights as it goes. Workspaces are
/// Qwen-Image 2.1's, measured with sd.cpp on Metal.
struct SDCppMemoryPlan {
    let textEncoder: Int64
    let transformer: Int64
    let vae: Int64
    let size: ImageSize
    /// Side of a decode tile in latent pixels; nil decodes the whole latent at once.
    let decodeTile: Int?
    /// The GPU budget sd.cpp runs under; nil lets it keep whole components resident.
    let gpuBudget: Int64?

    init(textEncoder: Int64, transformer: Int64, vae: Int64, size: ImageSize, decodeTile: Int?,
         gpuBudget: Int64?) {
        self.textEncoder = textEncoder
        self.transformer = transformer
        self.vae = vae
        self.size = size
        self.decodeTile = decodeTile
        self.gpuBudget = gpuBudget
    }

    init(files: SDCppModelFiles, size: ImageSize, decodeTile: Int?, gpuBudget: Int64?) {
        func bytes(_ url: URL) -> Int64 {
            ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? 0
        }
        self.init(textEncoder: bytes(files.textEncoder), transformer: bytes(files.diffusionModel),
                  vae: bytes(files.vae), size: size, decodeTile: decodeTile, gpuBudget: gpuBudget)
    }

    /// The denoiser's compute buffer grows with the image token count: 0.24 GB at 512 x 512,
    /// 0.95 GB at 1024 x 1024.
    static let denoiseBytesPerPixel: Double = 925
    /// The untiled VAE decode workspace grows with the output area: 2.04 GB at 512 x 512,
    /// 8.15 GB at 1024 x 1024.
    static let decodeBytesPerPixel: Double = 7_777
    /// Output pixels per latent pixel along each side.
    static let vaeScale = 16

    var denoiseWorkspace: Int64 { Int64(Self.denoiseBytesPerPixel * Double(size.width * size.height)) }

    var decodeWorkspace: Int64 {
        let untiled = Self.untiledDecodeWorkspace(size)
        guard let decodeTile else { return untiled }
        let side = Double(decodeTile * Self.vaeScale)
        return min(untiled, Int64(Self.decodeBytesPerPixel * side * side))
    }

    var peak: Int64 {
        func resident(_ weights: Int64) -> Int64 { gpuBudget.map { min(weights, $0) } ?? weights }
        return max(resident(textEncoder), resident(transformer) + denoiseWorkspace, resident(vae) + decodeWorkspace)
    }

    func capabilities(on device: DeviceTier) -> EngineCapabilities {
        let budget = device.memoryBudgetBytes
        let residency: EngineCapabilities.Residency = gpuBudget == nil ? .resident : .streamingInternal
        if peak <= Int64(Double(budget) * 0.9) {
            return EngineCapabilities(runnable: true, residency: residency, estimatedPeakBytes: peak,
                                      note: "Runs great")
        }
        if peak <= budget {
            return EngineCapabilities(runnable: true, residency: residency, estimatedPeakBytes: peak,
                                      note: "Tight fit")
        }
        return EngineCapabilities(runnable: false, residency: .unsupported, estimatedPeakBytes: peak,
                                  note: "Needs more memory")
    }

    static func untiledDecodeWorkspace(_ size: ImageSize) -> Int64 {
        Int64(decodeBytesPerPixel * Double(size.width * size.height))
    }

    /// Tile the decode once its untiled workspace would take more than 30% of the device's budget:
    /// tiling costs time (overlapping tiles are decoded twice), so it is kept for renders that need it.
    static func shouldTile(_ size: ImageSize, on device: DeviceTier) -> Bool {
        Double(untiledDecodeWorkspace(size)) > Double(device.memoryBudgetBytes) * 0.3
    }

    /// Tile side in latent pixels: sd.cpp's default 32 (512 px) on a Mac, 16 (256 px, a quarter of
    /// the workspace) on a phone.
    static func decodeTile(on device: DeviceTier) -> Int { device.isPhone ? 16 : 32 }
}
