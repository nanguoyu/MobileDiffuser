// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import CoreGraphics
import DiffusionCore
import Foundation
import os
import StableDiffusionCpp

/// Runs any model stable-diffusion.cpp supports (the first is Qwen-Image-2.1) behind the same
/// `DiffusionEngine` boundary as the MLX engines, so the app drives it exactly like them.
///
/// Like the FLUX facade, it resolves its own weights: it is constructed with the model's files and
/// ignores the `WeightSource` passed to `load`. sd.cpp owns everything below the C API (graph
/// construction, Metal kernels, memory mapping, segmenting the graph to a GPU budget), which is the
/// point of having it: no per-model port.
///
/// sd.cpp's progress and log callbacks are process-global. The app runs one generation at a time,
/// and every call into sd.cpp is funnelled through one serial queue, so they never overlap.
public actor SDCppDiffusionEngine: DiffusionEngine {

    private let files: SDCppModelFiles
    private let options: SDCppOptions
    private var context: SDContextHandle?

    public init(files: SDCppModelFiles, options: SDCppOptions = SDCppOptions()) {
        self.files = files
        self.options = options
    }

    public static func capabilities(for model: DiffusionModel,
                                    variant: ModelVariant,
                                    on device: DeviceTier) -> EngineCapabilities {
        capabilities(for: model, variant: variant, on: device, size: .square1024)
    }

    /// How a render of `size` fits `device`, phase by phase (see `SDCppMemoryPlan`), with the decode
    /// tiled and the GPU budget set the way the engine would run it there.
    public static func capabilities(for model: DiffusionModel, variant: ModelVariant,
                                    on device: DeviceTier, size: ImageSize) -> EngineCapabilities {
        let c = variant.components
        let tiled = SDCppMemoryPlan.shouldTile(size, on: device)
        return SDCppMemoryPlan(textEncoder: c.textEncoder, transformer: c.transformer, vae: c.vae, size: size,
                               decodeTile: tiled ? SDCppMemoryPlan.decodeTile(on: device) : nil,
                               gpuBudget: defaultGPUBudgetGiB(on: device).map { Int64($0 * 1_073_741_824) })
            .capabilities(on: device)
    }

    /// A phone's GPU shares its RAM with the whole system and the buffers sd.cpp keeps resident are
    /// wired, so there sd.cpp is held to 30% of the RAM: it then runs larger components in segments,
    /// reading their weights as it goes. On a Mac it sizes itself.
    static func defaultGPUBudgetGiB(on device: DeviceTier) -> Double? {
        device.isPhone ? Double(device.physicalMemoryBytes) * 0.3 / 1_073_741_824 : nil
    }

    private var gpuBudgetGiB: Double? { options.gpuBudgetGiB ?? Self.defaultGPUBudgetGiB(on: .current) }

    public func load(_ model: DiffusionModel,
                     variant: ModelVariant,
                     source: WeightSource,
                     progress: @Sendable @escaping (Double) -> Void) async throws {
        if context != nil { progress(1); return }
        for url in files.required where !FileManager.default.fileExists(atPath: url.path) {
            throw SDCppError.missingFile(url.lastPathComponent)
        }
        SDLog.installIfNeeded()
        let files = self.files, options = self.options, gpuBudgetGiB = self.gpuBudgetGiB
        let handle = try await Self.onSDQueue { () throws -> SDContextHandle in
            let relay = ProgressRelay(onLoad: progress)
            let strings = CStringPool()
            return try withExtendedLifetime((relay, strings)) {
                sd_set_progress_callback(progressTrampoline, Unmanaged.passUnretained(relay).toOpaque())
                defer { sd_set_progress_callback(nil, nil) }

                var params = sd_ctx_params_t()
                sd_ctx_params_init(&params)
                params.diffusion_model_path = strings.make(files.diffusionModel.path)
                params.llm_path = strings.make(files.textEncoder.path)
                params.vae_path = strings.make(files.vae.path)
                if let vision = files.textEncoderVision {
                    params.llm_vision_path = strings.make(vision.path)
                }
                params.n_threads = sd_get_num_physical_cores()
                params.enable_mmap = options.memoryMapWeights
                params.flash_attn = options.flashAttention
                params.diffusion_flash_attn = options.flashAttention
                if let gib = gpuBudgetGiB {
                    params.max_vram = strings.make(String(format: "%.2f", gib))
                }
                // sd.cpp copies every path into its own storage, so the pool can go once this returns.
                SDLog.shared.beginCall()
                guard let raw = new_sd_ctx(&params) else {
                    throw SDCppError.loadFailed(SDLog.shared.callError())
                }
                return SDContextHandle(raw)
            }
        }
        if context != nil {
            // Another load finished while this one ran (the actor is re-entrant); keep that one and
            // free this context on the sd.cpp queue, never on whichever thread drops it last.
            try? await Self.onSDQueue { handle.release() }
        } else {
            context = handle
        }
        progress(1)
    }

    public func generate(_ request: GenerationRequest,
                         progress: @Sendable @escaping (GenerationProgress) -> Void) async throws -> CGImage {
        guard let context else { throw SDCppError.notLoaded }
        if !request.referenceImages.isEmpty && files.textEncoderVision == nil {
            throw SDCppError.editingNeedsVisionEncoder
        }
        let references = try request.referenceImages.map { image -> SDInputImage in
            guard let input = SDInputImage(image) else { throw SDCppError.unreadableImage }
            return input
        }
        let device = DeviceTier.current
        let tiledDecode = options.tiledVAEDecode ?? SDCppMemoryPlan.shouldTile(request.size, on: device)
        let decodeTile = tiledDecode ? SDCppMemoryPlan.decodeTile(on: device) : nil
        #if os(iOS)
        // On iOS running out of memory ends the app, so a render that cannot fit is refused up front.
        guard SDCppMemoryPlan(files: files, size: request.size, decodeTile: decodeTile,
                              gpuBudget: gpuBudgetGiB.map { Int64($0 * 1_073_741_824) })
            .capabilities(on: device).runnable else { throw EngineError.unsupportedOnDevice }
        #endif
        let relay = ProgressRelay(onGenerate: progress, steps: request.steps, control: request.control) {
            sd_cancel_generation(context.raw, SD_CANCEL_ALL)
        }
        progress(.encoding)

        let outcome: GenerationOutcome = try await withTaskCancellationHandler {
            try await Self.onSDQueue { () -> GenerationOutcome in
                let strings = CStringPool()
                var rawReferences = references.map(\.raw)
                return withExtendedLifetime((relay, strings, references)) {
                    // A cancel that arrived after the previous run finished must not abort this one.
                    sd_cancel_generation(context.raw, SD_CANCEL_RESET)
                    SDLog.shared.beginCall()
                    sd_set_progress_callback(progressTrampoline, Unmanaged.passUnretained(relay).toOpaque())
                    defer { sd_set_progress_callback(nil, nil) }

                    var params = sd_img_gen_params_t()
                    sd_img_gen_params_init(&params)
                    params.prompt = strings.make(request.prompt)
                    params.negative_prompt = strings.make(request.negativePrompt ?? "")
                    params.width = Int32(request.size.width)
                    params.height = Int32(request.size.height)
                    // sd.cpp draws a random seed for negative values, so keep every seed non-negative.
                    params.seed = Int64(request.seed & UInt64(Int64.max))
                    params.batch_count = 1
                    let method = sd_get_default_sample_method(context.raw)
                    params.sample_params.sample_method = method
                    params.sample_params.scheduler = sd_get_default_scheduler(context.raw, method)
                    params.sample_params.sample_steps = Int32(request.steps)
                    // 1.0 means guidance-free: sd.cpp then skips the unconditional pass entirely.
                    params.sample_params.guidance.txt_cfg = request.guidance
                    params.vae_tiling_params.enabled = decodeTile != nil
                    if let decodeTile {
                        params.vae_tiling_params.tile_size_x = Int32(decodeTile)
                        params.vae_tiling_params.tile_size_y = Int32(decodeTile)
                    }

                    var images: UnsafeMutablePointer<sd_image_t>?
                    var count: Int32 = 0
                    let ok = rawReferences.withUnsafeMutableBufferPointer { buffer -> Bool in
                        params.ref_images = buffer.baseAddress
                        params.ref_images_count = Int32(buffer.count)
                        return generate_image(context.raw, &params, &images, &count)
                    }
                    defer { SDImage.free(images, count: Int(count)) }

                    if relay.wasCancelled { return .cancelled }
                    guard ok, count > 0, let first = images?.pointee,
                          let image = SDImage.makeCGImage(first) else {
                        return .failed(SDLog.shared.callError())
                    }
                    return .image(image)
                }
            }
        } onCancel: {
            // Also wake a run held at a paused checkpoint; the sd.cpp queue is not part of this task,
            // so cancelling the task alone would leave it waiting for a resume that never comes.
            request.control?.cancel()
            sd_cancel_generation(context.raw, SD_CANCEL_ALL)
        }

        if Task.isCancelled { throw CancellationError() }
        switch outcome {
        case .image(let image):
            progress(.finished(image))
            return image
        case .cancelled:
            throw CancellationError()
        case .failed(let detail):
            throw SDCppError.generationFailed(detail)
        }
    }

    public func unload() async {
        guard let context else { return }
        self.context = nil
        // Freed on the sd.cpp queue, so it can never race a generation still unwinding there.
        try? await Self.onSDQueue { context.release() }
    }

    // MARK: - The single sd.cpp queue

    private static let sdQueue = DispatchQueue(label: "MobileDiffuser.sdcpp", qos: .userInitiated)

    /// Every sd.cpp call is long and blocking; running them here keeps them off the cooperative
    /// thread pool and serialises them against each other.
    private static func onSDQueue<T>(_ work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            sdQueue.async { continuation.resume(with: Result(catching: work)) }
        }
    }
}

private enum GenerationOutcome {
    case image(CGImage)
    case cancelled
    case failed(String)
}

// MARK: - Context ownership

/// Owns one `sd_ctx_t`. Freed exactly once, explicitly or when the last reference goes away.
final class SDContextHandle: @unchecked Sendable {
    let raw: OpaquePointer
    private let lock = NSLock()
    private var freed = false

    init(_ raw: OpaquePointer) { self.raw = raw }

    func release() {
        lock.lock()
        defer { lock.unlock() }
        guard !freed else { return }
        freed = true
        free_sd_ctx(raw)
    }

    deinit { release() }
}

/// C strings that must outlive one sd.cpp call.
private final class CStringPool {
    private var storage: [UnsafeMutablePointer<CChar>] = []

    func make(_ string: String) -> UnsafePointer<CChar> {
        let copy = strdup(string)!
        storage.append(copy)
        return UnsafePointer(copy)
    }

    deinit { storage.forEach { free($0) } }
}

// MARK: - Progress

/// Receives sd.cpp's step callbacks on the sd.cpp queue and turns them into app progress. Also the
/// place where pause and cancel take effect: blocking here holds sd.cpp between two steps.
final class ProgressRelay: @unchecked Sendable {
    private let onLoad: (@Sendable (Double) -> Void)?
    private let onGenerate: (@Sendable (GenerationProgress) -> Void)?
    private let steps: Int
    private let control: GenerationControl?
    private let cancel: () -> Void

    // Touched only from the sd.cpp queue.
    private var denoised = false
    private(set) var wasCancelled = false

    init(onLoad: @escaping @Sendable (Double) -> Void) {
        self.onLoad = onLoad
        self.onGenerate = nil
        self.steps = 0
        self.control = nil
        self.cancel = {}
    }

    /// `cancel` stops sd.cpp; it runs on the sd.cpp queue when a checkpoint reports cancellation.
    init(onGenerate: @escaping @Sendable (GenerationProgress) -> Void, steps: Int,
         control: GenerationControl?, cancel: @escaping () -> Void) {
        self.onLoad = nil
        self.onGenerate = onGenerate
        self.steps = steps
        self.control = control
        self.cancel = cancel
    }

    func report(step: Int, of total: Int) {
        if let onLoad {
            if total > 0 { onLoad(min(1, Double(step) / Double(total))) }
            return
        }
        guard let onGenerate else { return }
        // sd.cpp reports several counters through this one callback: weight loading, sampler steps,
        // and (with a tiled VAE) decode tiles. The sampler's run is the one whose total equals the
        // requested step count, and decoding starts as soon as its last step completes.
        if !denoised && total == steps {
            onGenerate(.denoising(step: min(step, total), total: total, preview: nil))
            if step >= total {
                denoised = true
                onGenerate(.decoding)
            }
        }
        guard let control else { return }
        do {
            try control.checkpoint()
        } catch {
            if !wasCancelled { wasCancelled = true; cancel() }
        }
    }
}

private let progressTrampoline: sd_progress_cb_t = { step, steps, _, data in
    guard let data else { return }
    Unmanaged<ProgressRelay>.fromOpaque(data).takeUnretainedValue()
        .report(step: Int(step), of: Int(steps))
}

// MARK: - Logging

/// Forwards sd.cpp's log to the unified log, and keeps the reason the current call failed so a
/// failed load or generation can say why instead of failing silently.
private final class SDLog: @unchecked Sendable {
    static let shared = SDLog()
    private let logger = Logger(subsystem: "MobileDiffuser", category: "sd.cpp")
    private let lock = NSLock()
    /// The first warning or error of the call that says something failed: sd.cpp reports the cause
    /// (for instance which file could not be read) before the error that ends the call.
    private var firstFailure = ""
    private var firstError = ""
    private var installed = false

    static func installIfNeeded() {
        shared.lock.lock()
        defer { shared.lock.unlock() }
        guard !shared.installed else { return }
        shared.installed = true
        sd_set_log_callback(logTrampoline, nil)
    }

    func record(_ level: sd_log_level_t, _ text: String) {
        let line = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return }
        switch level {
        case SD_LOG_ERROR:
            logger.error("\(line, privacy: .public)")
            note(line, isError: true)
        case SD_LOG_WARN:
            logger.warning("\(line, privacy: .public)")
            note(line, isError: false)
        default:
            logger.debug("\(line, privacy: .public)")
        }
    }

    private func note(_ line: String, isError: Bool) {
        lock.lock(); defer { lock.unlock() }
        if isError && firstError.isEmpty { firstError = Self.readable(line) }
        if firstFailure.isEmpty && line.localizedCaseInsensitiveContains("fail") { firstFailure = Self.readable(line) }
    }

    /// Starts a call into sd.cpp: a problem logged by an earlier call must not become this one's reason.
    func beginCall() {
        lock.lock(); firstFailure = ""; firstError = ""; lock.unlock()
    }

    /// Why the current call failed, or "" when sd.cpp logged nothing that says.
    func callError() -> String {
        lock.lock(); defer { lock.unlock() }
        return firstFailure.isEmpty ? firstError : firstFailure
    }

    /// Drops sd.cpp's "file.cpp:123 - " origin and shortens quoted paths ('/…/models/x.gguf') to the
    /// file name: the reason is shown to people.
    static func readable(_ line: String) -> String {
        line.replacingOccurrences(of: "^[A-Za-z0-9_.]+:[0-9]+ *- *", with: "", options: .regularExpression)
            .replacingOccurrences(of: "'[^']*/([^'/]+)'", with: "$1", options: .regularExpression)
    }
}

private let logTrampoline: sd_log_cb_t = { level, text, _ in
    guard let text else { return }
    SDLog.shared.record(level, String(cString: text))
}
