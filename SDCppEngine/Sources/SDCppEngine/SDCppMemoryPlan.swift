// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import DiffusionCore
import Foundation

/// How much memory one render needs, phase by phase.
///
/// The weights are memory-mapped, so a phase needs only its own component resident, and the page
/// cache can take the previous phase's weights back: encoding holds the text encoder, denoising the
/// denoiser plus its compute buffer, decoding the VAE plus its workspace. The peak is the largest
/// phase. Workspaces are Qwen-Image 2.1's, measured with sd.cpp on Metal.
struct SDCppMemoryPlan {
    let textEncoder: Int64
    let transformer: Int64
    let vae: Int64
    let size: ImageSize
    let tiledDecode: Bool

    init(textEncoder: Int64, transformer: Int64, vae: Int64, size: ImageSize, tiledDecode: Bool) {
        self.textEncoder = textEncoder
        self.transformer = transformer
        self.vae = vae
        self.size = size
        self.tiledDecode = tiledDecode
    }

    init(files: SDCppModelFiles, size: ImageSize, tiledDecode: Bool) {
        func bytes(_ url: URL) -> Int64 {
            ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? 0
        }
        self.init(textEncoder: bytes(files.textEncoder), transformer: bytes(files.diffusionModel),
                  vae: bytes(files.vae), size: size, tiledDecode: tiledDecode)
    }

    /// The denoiser's compute buffer grows with the image token count: 0.24 GB at 512 x 512,
    /// 0.95 GB at 1024 x 1024.
    static let denoiseBytesPerPixel: Double = 925
    /// The untiled VAE decode workspace grows with the output area: 2.04 GB at 512 x 512,
    /// 8.15 GB at 1024 x 1024.
    static let decodeBytesPerPixel: Double = 7_777
    /// A tiled decode works on 512 x 512 tiles, so its workspace stays that of one tile.
    static let tileDecodeBytes: Int64 = 2_040_000_000

    var denoiseWorkspace: Int64 { Int64(Self.denoiseBytesPerPixel * Double(size.width * size.height)) }

    var decodeWorkspace: Int64 {
        let untiled = Self.untiledDecodeWorkspace(size)
        return tiledDecode ? min(untiled, Self.tileDecodeBytes) : untiled
    }

    var peak: Int64 { max(textEncoder, transformer + denoiseWorkspace, vae + decodeWorkspace) }

    func capabilities(on device: DeviceTier) -> EngineCapabilities {
        let budget = device.memoryBudgetBytes
        if peak <= Int64(Double(budget) * 0.9) {
            return EngineCapabilities(runnable: true, residency: .resident, estimatedPeakBytes: peak,
                                      note: "Runs great")
        }
        if peak <= budget {
            return EngineCapabilities(runnable: true, residency: .resident, estimatedPeakBytes: peak,
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
}
