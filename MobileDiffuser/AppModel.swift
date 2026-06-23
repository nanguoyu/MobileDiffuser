// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import CoreGraphics
import ZImageMLX

/// Drives model download (via `ModelDownloader`) and generation (via `ZImagePipeline`). UI state
/// lives on the main actor; the heavy denoise runs on a detached task so the UI stays responsive.
@MainActor
@Observable
final class AppModel {
    enum Phase: Equatable {
        case idle
        case downloading(Double)    // 0…1 model download
        case loading(Double)        // 0…1 weight load into MLX
        case generating(Int, Int)   // step, total
        case done
        case failed(String)
    }

    let models = ZImageCatalog.all
    var selected: ZImageCatalogModel = ZImageCatalog.turboQ4
    var prompt = "a red panda on a mossy rock, soft morning light"
    var size = 1024
    var steps = 8
    var seedText = "42"
    var phase: Phase = .idle
    var image: CGImage?

    private let downloader: ModelDownloader
    private var pipeline: ZImagePipeline?
    private var loadedDirectory: String?

    init() {
        let support = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let base = (support ?? URL(fileURLWithPath: NSTemporaryDirectory())).appending(component: "MobileDiffuser")
        downloader = ModelDownloader(downloadBase: base)
    }

    var isDownloaded: Bool { downloader.isDownloaded(repoId: selected.id) }

    var isBusy: Bool {
        switch phase { case .downloading, .loading, .generating: return true; default: return false }
    }

    var isFailed: Bool { if case .failed = phase { return true } else { return false } }

    var statusText: String {
        switch phase {
        case .idle: return isDownloaded ? "Ready" : "Not downloaded"
        case .downloading(let f): return "Downloading model… \(Int(f * 100))%"
        case .loading(let f): return "Loading into memory… \(Int(f * 100))%"
        case .generating(let s, let t): return "Generating… step \(s)/\(t)"
        case .done: return "Done"
        case .failed(let m): return "Failed: \(m)"
        }
    }

    /// Download the selected model (idempotent — fast if already present).
    func download() async {
        do {
            phase = .downloading(0)
            _ = try await downloader.download(repoId: selected.id) { fraction in
                Task { @MainActor in self.phase = .downloading(fraction) }
            }
            phase = .idle
        } catch {
            phase = .failed(String(describing: error))
        }
    }

    /// Generate an image, downloading + loading the model first if needed.
    func generate() async {
        if !isDownloaded {
            await download()
            guard isDownloaded else { return }
        }
        let directory = downloader.localURL(repoId: selected.id).path
        let prompt = self.prompt, size = self.size, steps = self.steps
        let seed = UInt64(seedText) ?? 42
        do {
            if pipeline == nil || loadedDirectory != directory {
                phase = .loading(0)
                let loaded = ZImagePipeline(modelDirectory: URL(fileURLWithPath: directory))
                try await loaded.loadModels { fraction in
                    Task { @MainActor in self.phase = .loading(fraction) }
                }
                pipeline = loaded
                loadedDirectory = directory
            }
            guard let pipeline else { return }
            phase = .generating(0, steps)
            let cgImage = try await Task.detached(priority: .userInitiated) {
                try pipeline.generate(prompt: prompt, size: size, steps: steps, seed: seed) { step, total in
                    Task { @MainActor in self.phase = .generating(step, total) }
                }
            }.value
            image = cgImage
            phase = .done
        } catch {
            phase = .failed(String(describing: error))
        }
    }
}
