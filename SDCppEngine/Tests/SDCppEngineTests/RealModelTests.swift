// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import CoreGraphics
import DiffusionCore
import Foundation
import XCTest
@testable import SDCppEngine

/// Runs the real Qwen-Image 2.1 GGUF model through the engine. Opt-in: set `SDCPP_TEST_MODELS` to a
/// folder holding qwen-image-2.1-Q4_K_M.gguf, Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf and
/// qwen_image_2.1_vae_bf16.safetensors (with xcodebuild, export TEST_RUNNER_SDCPP_TEST_MODELS).
final class RealModelTests: XCTestCase {

    func testGeneratesAnImageAndReportsEveryStep() async throws {
        let engine = SDCppDiffusionEngine(files: try Self.files())
        try await engine.load(Self.model, variant: Self.model.variants[0],
                              source: SafetensorsWeightSource(tensors: [:]), progress: { _ in })
        let events = EventRecorder()

        let image = try await engine.generate(Self.request(steps: 2)) { events.record($0) }
        await engine.unload()

        XCTAssertEqual(image.width, 256)
        XCTAssertEqual(image.height, 256)
        XCTAssertEqual(events.labels.first, "encoding")
        XCTAssertEqual(events.labels.filter { $0.hasPrefix("denoising") },
                       ["denoising 0/2", "denoising 1/2", "denoising 2/2"])
        XCTAssertEqual(Array(events.labels.suffix(2)), ["decoding", "finished"])
        XCTAssertGreaterThan(Self.pixelSpread(image), 20, "a real render is not a flat image")
    }

    func testCancelStopsTheRunAndTheEngineStaysUsable() async throws {
        let engine = SDCppDiffusionEngine(files: try Self.files())
        try await engine.load(Self.model, variant: Self.model.variants[0],
                              source: SafetensorsWeightSource(tensors: [:]), progress: { _ in })
        let control = GenerationControl()
        let started = Date()
        do {
            _ = try await engine.generate(Self.request(steps: 30, control: control)) { progress in
                if case .denoising(1, _, _) = progress { control.cancel() }
            }
            XCTFail("a cancelled run must not return an image")
        } catch is CancellationError {
        }
        XCTAssertLessThan(Date().timeIntervalSince(started), 60, "cancel takes effect at the next step")

        // sd.cpp's cancel flag is sticky; the next run must start clean.
        let image = try await engine.generate(Self.request(steps: 1)) { _ in }
        await engine.unload()
        XCTAssertEqual(image.width, 256)
    }

    // MARK: - Helpers

    private static func files() throws -> SDCppModelFiles {
        guard let folder = ProcessInfo.processInfo.environment["SDCPP_TEST_MODELS"], !folder.isEmpty else {
            throw XCTSkip("SDCPP_TEST_MODELS is not set")
        }
        let dir = URL(fileURLWithPath: folder)
        return SDCppModelFiles(diffusionModel: dir.appendingPathComponent("qwen-image-2.1-Q4_K_M.gguf"),
                               textEncoder: dir.appendingPathComponent("Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"),
                               vae: dir.appendingPathComponent("qwen_image_2.1_vae_bf16.safetensors"))
    }

    private static func request(steps: Int, control: GenerationControl? = nil) -> GenerationRequest {
        GenerationRequest(prompt: "a red apple on a wooden table", steps: steps, guidance: 1, seed: 3,
                          size: ImageSize(width: 256, height: 256), control: control)
    }

    /// Max minus min over the image's bytes: a flat or failed decode has almost none.
    private static func pixelSpread(_ image: CGImage) -> Int {
        guard let data = image.dataProvider?.data as Data?, let low = data.min(), let high = data.max() else { return 0 }
        return Int(high) - Int(low)
    }

    private static let model = DiffusionModel(
        id: "qwen-image-2.1-gguf", displayName: "Qwen-Image 2.1", family: .qwenImage, publisher: "Qwen",
        summary: "", license: .other(name: "Qwen Research", commercialUse: false),
        architecture: ArchitectureSpec(family: .qwenImage, latentChannels: 64,
                                       defaultSampler: .flowMatchEuler, defaultSteps: 20,
                                       defaultGuidance: 1),
        variants: [ModelVariant(precision: .q4, approximateBytes: 1,
                                components: ComponentSizes(transformer: 1, textEncoder: 1, vae: 1),
                                layout: .flatSingle,
                                source: ModelSource(huggingFaceRepo: "unsloth/Qwen-Image-2.1-GGUF"))])
}

private final class EventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [String] = []

    func record(_ progress: GenerationProgress) {
        let label: String
        switch progress {
        case .denoising(let step, let total, _): label = "denoising \(step)/\(total)"
        case .decoding: label = "decoding"
        case .encoding: label = "encoding"
        case .preparing: label = "preparing"
        case .downloading: label = "downloading"
        case .cooling: label = "cooling"
        case .finished: label = "finished"
        }
        lock.lock(); events.append(label); lock.unlock()
    }

    var labels: [String] { lock.lock(); defer { lock.unlock() }; return events }
}
