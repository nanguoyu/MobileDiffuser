// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import Foundation

/// The files Qwen-Image 2.1 is assembled from and the quantizations the app offers. The denoiser and
/// the text encoder each come as one GGUF file per quantization, published side by side, so only the
/// chosen file of each is downloaded; the VAE is one bf16 safetensors file.
enum QwenImage21Files {
    static let transformerRepo = "unsloth/Qwen-Image-2.1-GGUF"
    static let encoderRepo = "unsloth/Qwen3-VL-8B-Instruct-GGUF"
    static let vaeRepo = "Comfy-Org/Qwen-Image-2.1"

    /// One downloadable file, identified by the recipe component id.
    struct File: Hashable, Sendable {
        let id: String
        let repo: String
        /// Path inside the repository (also its path under the repository's local folder).
        let path: String
        let bytes: Int64
    }

    enum Transformer: String, CaseIterable, Sendable {
        case q3 = "Q3_K_M", q4 = "Q4_K_M", q5 = "Q5_K_M", q6 = "Q6_K", q8 = "Q8_0"

        var label: String { rawValue }
        var note: String {
            switch self {
            case .q3: "smallest, 3.2 GB"
            case .q4: "recommended, 4.2 GB"
            case .q5: "sharper detail, 5.4 GB"
            case .q6: "near lossless, 6.3 GB"
            case .q8: "best quality, 7.6 GB"
            }
        }
        var file: File {
            let bytes: Int64 = switch self {
            case .q3: 3_168_290_528
            case .q4: 4_199_565_024
            case .q5: 5_390_223_072
            case .q6: 6_271_551_200
            case .q8: 7_640_860_384
            }
            return File(id: "qwen21-dit-\(rawValue)", repo: QwenImage21Files.transformerRepo,
                        path: "qwen-image-2.1-\(rawValue).gguf", bytes: bytes)
        }
    }

    enum Encoder: String, CaseIterable, Sendable {
        case q3 = "UD-Q3_K_XL", q4 = "UD-Q4_K_XL", q8 = "Q8_0"

        var label: String { rawValue }
        var note: String {
            switch self {
            case .q3: "smallest, 4.3 GB"
            case .q4: "recommended, 5.1 GB"
            case .q8: "best prompt fidelity, 8.7 GB"
            }
        }
        var file: File {
            let bytes: Int64 = switch self {
            case .q3: 4_313_344_864
            case .q4: 5_148_699_488
            case .q8: 8_709_520_224
            }
            return File(id: "qwen21-te-\(rawValue)", repo: QwenImage21Files.encoderRepo,
                        path: "Qwen3-VL-8B-Instruct-\(rawValue).gguf", bytes: bytes)
        }
    }

    static let vae = File(id: "qwen21-vae", repo: vaeRepo,
                          path: "vae/qwen_image_2.1_vae_bf16.safetensors", bytes: 675_509_688)

    /// The files one recipe runs.
    static func active(_ transformer: Transformer, _ encoder: Encoder) -> [File] {
        [transformer.file, encoder.file, vae]
    }

    /// Every file the details screen can install, in display order.
    static var all: [File] {
        Transformer.allCases.map(\.file) + Encoder.allCases.map(\.file) + [vae]
    }
}
