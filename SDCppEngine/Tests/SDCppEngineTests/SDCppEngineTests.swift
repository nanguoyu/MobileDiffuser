// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import CoreGraphics
import DiffusionCore
import Foundation
import StableDiffusionCpp
import XCTest
@testable import SDCppEngine

final class ProgressRelayTests: XCTestCase {

    func testSamplerStepsBecomeDenoisingThenOneDecoding() {
        let events = EventLog()
        let relay = ProgressRelay(onGenerate: { events.append($0) }, steps: 4, control: nil) {}

        relay.report(step: 120, of: 265)          // lazy weight loading before sampling: not a step
        for step in 0...4 { relay.report(step: step, of: 4) }
        for tile in [0, 8, 16] { relay.report(step: tile, of: 16) }   // tiled VAE decode

        XCTAssertEqual(events.labels, ["denoising 0/4", "denoising 1/4", "denoising 2/4",
                                       "denoising 3/4", "denoising 4/4", "decoding"])
    }

    func testTileCountEqualToStepCountIsStillDecoding() {
        let events = EventLog()
        let relay = ProgressRelay(onGenerate: { events.append($0) }, steps: 2, control: nil) {}
        for step in 0...2 { relay.report(step: step, of: 2) }
        for tile in 0...2 { relay.report(step: tile, of: 2) }
        XCTAssertEqual(events.labels, ["denoising 0/2", "denoising 1/2", "denoising 2/2", "decoding"])
    }

    func testCancelledControlStopsSDCppOnce() {
        let control = GenerationControl()
        let cancels = Counter()
        let relay = ProgressRelay(onGenerate: { _ in }, steps: 4, control: control) { cancels.increment() }

        relay.report(step: 1, of: 4)
        XCTAssertFalse(relay.wasCancelled)
        control.cancel()
        relay.report(step: 2, of: 4)
        relay.report(step: 3, of: 4)

        XCTAssertTrue(relay.wasCancelled)
        XCTAssertEqual(cancels.value, 1)
    }

    func testPauseHoldsTheStepUntilResumed() {
        let control = GenerationControl()
        let relay = ProgressRelay(onGenerate: { _ in }, steps: 4, control: control) {}
        control.pause()

        let returned = expectation(description: "the step returns after resume")
        let finished = Counter()
        Thread {
            relay.report(step: 1, of: 4)
            finished.increment()
            returned.fulfill()
        }.start()

        Thread.sleep(forTimeInterval: 0.2)
        XCTAssertEqual(finished.value, 0, "a paused run must not move past the step")
        control.resume()
        wait(for: [returned], timeout: 5)
        XCTAssertFalse(relay.wasCancelled)
    }

    func testCancelWhilePausedReleasesTheStep() {
        let control = GenerationControl()
        let cancels = Counter()
        let relay = ProgressRelay(onGenerate: { _ in }, steps: 4, control: control) { cancels.increment() }
        control.pause()

        let returned = expectation(description: "the step returns after cancel")
        Thread {
            relay.report(step: 1, of: 4)
            returned.fulfill()
        }.start()

        Thread.sleep(forTimeInterval: 0.1)
        control.cancel()
        wait(for: [returned], timeout: 5)
        XCTAssertTrue(relay.wasCancelled)
        XCTAssertEqual(cancels.value, 1)
    }

    func testLoadProgressIsAFraction() {
        let fractions = FractionLog()
        let relay = ProgressRelay(onLoad: { fractions.append($0) })
        relay.report(step: 0, of: 0)
        relay.report(step: 50, of: 200)
        relay.report(step: 200, of: 200)
        XCTAssertEqual(fractions.values, [0.25, 1.0])
    }
}

final class SDImageTests: XCTestCase {

    func testRGBPixelsSurviveTheCopy() throws {
        var pixels: [UInt8] = [255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30]   // 2x2 RGB
        let image = try pixels.withUnsafeMutableBufferPointer { buffer in
            try XCTUnwrap(SDImage.makeCGImage(sd_image_t(width: 2, height: 2, channel: 3,
                                                         data: buffer.baseAddress)))
        }
        pixels = [UInt8](repeating: 0, count: 12)   // the image must not borrow the source buffer

        XCTAssertEqual(image.width, 2)
        XCTAssertEqual(image.alphaInfo, .none)
        XCTAssertEqual(try rgba(of: image), [255, 0, 0, 255, 0, 255, 0, 255,
                                             0, 0, 255, 255, 10, 20, 30, 255])
    }

    func testRGBAKeepsStraightAlpha() throws {
        var pixels: [UInt8] = [200, 100, 50, 128]
        let image = try pixels.withUnsafeMutableBufferPointer { buffer in
            try XCTUnwrap(SDImage.makeCGImage(sd_image_t(width: 1, height: 1, channel: 4,
                                                         data: buffer.baseAddress)))
        }
        XCTAssertEqual(image.alphaInfo, .last)
        let bytes = try XCTUnwrap(image.dataProvider?.data as Data?)
        XCTAssertEqual([UInt8](bytes), [200, 100, 50, 128])
    }

    func testUnsupportedChannelCountIsRejected() {
        var pixels: [UInt8] = [1, 2]
        let image = pixels.withUnsafeMutableBufferPointer {
            SDImage.makeCGImage(sd_image_t(width: 1, height: 1, channel: 2, data: $0.baseAddress))
        }
        XCTAssertNil(image)
    }

    func testFreeReleasesEveryBuffer() {
        let images = UnsafeMutablePointer<sd_image_t>.allocate(capacity: 2)
        for i in 0..<2 {
            let data = UnsafeMutablePointer<UInt8>.allocate(capacity: 3)
            images[i] = sd_image_t(width: 1, height: 1, channel: 3, data: data)
        }
        // Under ASan/leaks this is the check; here it must simply not crash or double-free.
        SDImage.free(images, count: 2)
        SDImage.free(nil, count: 0)
    }

    func testInputImageWithAlphaIsUnpremultiplied() throws {
        let source = try makeImage(width: 1, height: 1, rgba: [200, 100, 50, 128])
        let input = try XCTUnwrap(SDInputImage(source))
        XCTAssertEqual(input.channels, 4)
        let raw = input.raw
        let bytes = Array(UnsafeBufferPointer(start: raw.data, count: 4))
        XCTAssertEqual(Int(bytes[3]), 128)
        for (value, expected) in zip(bytes.prefix(3), [200, 100, 50]) {
            XCTAssertEqual(Int(value), expected, accuracy: 2)
        }
    }

    func testOpaqueInputImageIsRGB() throws {
        let source = try makeImage(width: 2, height: 1, rgba: [9, 8, 7, 255, 1, 2, 3, 255], opaque: true)
        let input = try XCTUnwrap(SDInputImage(source))
        XCTAssertEqual(input.channels, 3)
        let raw = input.raw
        XCTAssertEqual(Array(UnsafeBufferPointer(start: raw.data, count: 6)), [9, 8, 7, 1, 2, 3])
    }
}

final class MemoryPlanTests: XCTestCase {
    private let mac32 = DeviceTier(physicalMemoryBytes: 34_359_738_368, isPhone: false)
    private let mac16 = DeviceTier(physicalMemoryBytes: 17_179_869_184, isPhone: false)
    private let phone8 = DeviceTier(physicalMemoryBytes: 8_589_934_592, isPhone: true)
    private let square512 = ImageSize(width: 512, height: 512)

    /// Q4_K_M denoiser, UD-Q4_K_XL encoder, bf16 VAE.
    private func plan(_ size: ImageSize, tile: Int? = nil, budget: Int64? = nil) -> SDCppMemoryPlan {
        SDCppMemoryPlan(textEncoder: 5_148_699_488, transformer: 4_199_565_024, vae: 675_509_688,
                        size: size, decodeTile: tile, gpuBudget: budget)
    }

    /// The plan the engine would run on `device`.
    private func caps(_ device: DeviceTier, _ size: ImageSize) -> EngineCapabilities {
        let tile = SDCppMemoryPlan.shouldTile(size, on: device) ? SDCppMemoryPlan.decodeTile(on: device) : nil
        let budget = SDCppDiffusionEngine.defaultGPUBudgetGiB(on: device).map { Int64($0 * 1_073_741_824) }
        return plan(size, tile: tile, budget: budget).capabilities(on: device)
    }

    func testPeakIsTheLargestPhaseNotTheSumOfWeights() {
        XCTAssertEqual(plan(square512).peak, 5_148_699_488, "at 512 the text encoder phase dominates")
        let large = plan(.square1024)
        XCTAssertEqual(large.peak, 675_509_688 + SDCppMemoryPlan.untiledDecodeWorkspace(.square1024),
                       "an untiled 1024 decode dominates")
        XCTAssertLessThan(plan(.square1024, tile: 32).peak, large.peak)
    }

    func testSmallerTilesShrinkTheDecodeWorkspace() {
        XCTAssertEqual(plan(square512, tile: 32).decodeWorkspace, SDCppMemoryPlan.untiledDecodeWorkspace(square512),
                       "one 512 px tile covers a 512 px image")
        XCTAssertEqual(plan(square512, tile: 16).decodeWorkspace * 4,
                       SDCppMemoryPlan.untiledDecodeWorkspace(square512), accuracy: 4)
    }

    func testTheGPUBudgetCapsWhatStaysResident() {
        let budget: Int64 = 2_500_000_000
        let streamed = plan(.square1024, tile: 16, budget: budget)
        XCTAssertEqual(streamed.peak, budget + streamed.denoiseWorkspace, "the denoiser runs in segments")
        XCTAssertLessThan(streamed.peak, plan(.square1024, tile: 16).peak)
    }

    func testTilingIsKeptForRendersThatNeedIt() {
        XCTAssertFalse(SDCppMemoryPlan.shouldTile(square512, on: mac16))
        XCTAssertTrue(SDCppMemoryPlan.shouldTile(.square1024, on: mac16))
        XCTAssertFalse(SDCppMemoryPlan.shouldTile(.square1024, on: mac32))
        XCTAssertTrue(SDCppMemoryPlan.shouldTile(.square1024, on: phone8))
    }

    func testFitFollowsTheDevice() {
        XCTAssertTrue(caps(mac32, .square1024).runnable)
        XCTAssertTrue(caps(mac16, .square1024).runnable)
        XCTAssertTrue(caps(phone8, square512).runnable, "a phone runs it in segments under its GPU budget")
        XCTAssertFalse(plan(square512, tile: 16).capabilities(on: phone8).runnable,
                       "held resident, the 5 GB encoder alone is past a phone's budget")
    }
}

final class SDCppDiffusionEngineTests: XCTestCase {

    func testMissingFileIsReportedBeforeLoading() async {
        let directory = FileManager.default.temporaryDirectory
        let engine = SDCppDiffusionEngine(files: SDCppModelFiles(
            diffusionModel: directory.appendingPathComponent("absent-dit.gguf"),
            textEncoder: directory.appendingPathComponent("absent-te.gguf"),
            vae: directory.appendingPathComponent("absent-vae.safetensors")))
        do {
            try await engine.load(Self.model, variant: Self.model.variants[0], source: SafetensorsWeightSource(tensors: [:]),
                                  progress: { _ in })
            XCTFail("load must fail")
        } catch {
            XCTAssertEqual(error as? SDCppError, .missingFile("absent-dit.gguf"))
        }
    }

    func testGenerateBeforeLoadFails() async {
        let directory = FileManager.default.temporaryDirectory
        let engine = SDCppDiffusionEngine(files: SDCppModelFiles(
            diffusionModel: directory.appendingPathComponent("a.gguf"),
            textEncoder: directory.appendingPathComponent("b.gguf"),
            vae: directory.appendingPathComponent("c.safetensors")))
        do {
            _ = try await engine.generate(GenerationRequest(prompt: "a cat", steps: 2, seed: 1),
                                          progress: { _ in })
            XCTFail("generate must fail")
        } catch {
            XCTAssertEqual(error as? SDCppError, .notLoaded)
        }
    }

    /// Exercises the linked library end to end: sd.cpp must reject files that are not models and
    /// the engine must surface its reason instead of crashing or failing silently.
    func testCorruptModelFilesFailToLoadWithAReason() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("sdcpp-corrupt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let files = SDCppModelFiles(diffusionModel: directory.appendingPathComponent("dit.gguf"),
                                    textEncoder: directory.appendingPathComponent("te.gguf"),
                                    vae: directory.appendingPathComponent("vae.safetensors"))
        for url in files.required {
            try Data(repeating: 0x5A, count: 4096).write(to: url)
        }

        let engine = SDCppDiffusionEngine(files: files)
        do {
            try await engine.load(Self.model, variant: Self.model.variants[0], source: SafetensorsWeightSource(tensors: [:]),
                                  progress: { _ in })
            XCTFail("load must fail")
        } catch SDCppError.loadFailed(let detail) {
            XCTAssertTrue(detail.contains("dit.gguf"), "the reason names the file that failed: \(detail)")
            XCTAssertFalse(detail.contains("/"), "file names, not full paths: \(detail)")
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    private static let model = DiffusionModel(
        id: "test", displayName: "Test", family: .qwenImage, publisher: "Test", summary: "",
        license: .apache2,
        architecture: ArchitectureSpec(family: .qwenImage, latentChannels: 64,
                                       defaultSampler: .flowMatchEuler, defaultSteps: 20,
                                       defaultGuidance: 4),
        variants: [ModelVariant(precision: .q4, approximateBytes: 1,
                                components: ComponentSizes(transformer: 1, textEncoder: 1, vae: 1),
                                layout: .flatSingle,
                                source: ModelSource(huggingFaceRepo: "test/test"))])
}

// MARK: - Helpers

private final class EventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [String] = []

    func append(_ progress: GenerationProgress) {
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

private final class FractionLog: @unchecked Sendable {
    private let lock = NSLock()
    private var fractions: [Double] = []
    func append(_ value: Double) { lock.lock(); fractions.append(value); lock.unlock() }
    var values: [Double] { lock.lock(); defer { lock.unlock() }; return fractions }
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func increment() { lock.lock(); count += 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
}

private func makeImage(width: Int, height: Int, rgba: [UInt8], opaque: Bool = false) throws -> CGImage {
    let info = opaque ? CGImageAlphaInfo.noneSkipLast : CGImageAlphaInfo.premultipliedLast
    var pixels = rgba
    if !opaque {
        // CoreGraphics stores premultiplied colour; build the bytes a real premultiplied image would hold.
        for p in 0..<(width * height) {
            let a = Int(pixels[p * 4 + 3])
            for c in 0..<3 { pixels[p * 4 + c] = UInt8(Int(pixels[p * 4 + c]) * a / 255) }
        }
    }
    let provider = try XCTUnwrap(CGDataProvider(data: Data(pixels) as CFData))
    return try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                 bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                 bitmapInfo: CGBitmapInfo(rawValue: info.rawValue), provider: provider,
                                 decode: nil, shouldInterpolate: false, intent: .defaultIntent))
}

private func rgba(of image: CGImage) throws -> [UInt8] {
    var out = [UInt8](repeating: 0, count: image.width * image.height * 4)
    let drawn: Bool = out.withUnsafeMutableBytes { buffer in
        guard let context = CGContext(data: buffer.baseAddress, width: image.width, height: image.height,
                                      bitsPerComponent: 8, bytesPerRow: image.width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return false }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return true
    }
    XCTAssertTrue(drawn)
    return out
}
