// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Z-Image creation studio (dark theme): a model bar (download/manage) + a result canvas + a
/// prompt/controls panel. Runs `ZImagePipeline` via `AppModel`. First non-FLUX model on the MLX stack.
struct ContentView: View {
    @State private var model = AppModel()

    var body: some View {
        ZStack {
            Color(white: 0.06).ignoresSafeArea()
            VStack(spacing: 0) {
                modelBar
                canvas
                controls
            }
        }
        .tint(.orange)
        .preferredColorScheme(.dark)
    }

    private var modelBar: some View {
        HStack(spacing: 12) {
            Image(systemName: "cube.box.fill").foregroundStyle(.orange)
            Picker("Model", selection: $model.selectedID) {
                ForEach(model.models) { Text($0.displayName).tag($0.id) }
            }
            .pickerStyle(.menu).labelsHidden().disabled(model.isBusy)
            Text(model.selected.summary).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            Spacer()
            modelStatus
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(Color(white: 0.09))
    }

    @ViewBuilder private var modelStatus: some View {
        if case .downloading(let f) = model.phase {
            HStack(spacing: 6) {
                ProgressView(value: f).frame(width: 90)
                Text("\(Int(f * 100))%").font(.caption2).monospacedDigit().foregroundStyle(.secondary)
            }
        } else if model.managesOwnDownload {
            Label("Downloads on first run", systemImage: "icloud.and.arrow.down")
                .font(.caption2).foregroundStyle(.secondary)
        } else if model.isDownloaded {
            Label("Ready", systemImage: "checkmark.circle.fill").font(.caption).foregroundStyle(.green)
        } else {
            Button {
                Task { await model.download() }
            } label: {
                Label("Download · \(sizeString)", systemImage: "arrow.down.circle")
            }
            .buttonStyle(.bordered).controlSize(.small).disabled(model.isBusy)
        }
    }

    private var canvas: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 20, style: .continuous).fill(Color(white: 0.10))
            if let cg = model.image {
                Image(decorative: cg, scale: 1)
                    .resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            } else {
                VStack(spacing: 14) {
                    Image(systemName: model.isBusy ? "sparkles" : "photo.artframe")
                        .font(.system(size: 40, weight: .light)).foregroundStyle(.secondary)
                        .symbolEffect(.pulse, isActive: model.isBusy)
                    Text(model.isBusy ? model.statusText : "Describe an image, then Generate")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
            if case .generating(let step, let total) = model.phase {
                VStack {
                    Spacer()
                    ProgressView(value: Double(step), total: Double(total)).tint(.orange).padding()
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(16)
    }

    private var controls: some View {
        VStack(spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                TextField("Prompt", text: $model.prompt, axis: .vertical)
                    .textFieldStyle(.plain).lineLimit(1...3)
                    .padding(12)
                    .background(Color(white: 0.12), in: RoundedRectangle(cornerRadius: 12))
                Button {
                    Task { await model.generate() }
                } label: {
                    Image(systemName: "arrow.up").font(.headline).frame(width: 44, height: 44)
                }
                .buttonStyle(.borderedProminent).clipShape(Circle())
                .disabled(model.isBusy || model.prompt.isEmpty)
            }
            HStack(spacing: 10) {
                picker("Size", selection: $model.size, options: [512, 768, 1024]) { "\($0)" }
                picker("Steps", selection: $model.steps, options: [4, 8, 16]) { "\($0)" }
                TextField("Seed", text: $model.seedText).frame(width: 70).textFieldStyle(.roundedBorder)
                Spacer()
                Text(model.statusText)
                    .font(.caption).foregroundStyle(model.isFailed ? .red : .secondary).lineLimit(1)
            }
        }
        .padding(16)
        .background(Color(white: 0.09))
    }

    private var sizeString: String {
        ByteCountFormatter.string(fromByteCount: model.selected.variants[0].approximateBytes, countStyle: .file)
    }

    private func picker(_ label: String, selection: Binding<Int>, options: [Int],
                        format: @escaping (Int) -> String) -> some View {
        Picker(label, selection: selection) {
            ForEach(options, id: \.self) { Text(format($0)).tag($0) }
        }
        .pickerStyle(.menu).tint(.secondary)
    }
}

#Preview {
    ContentView()
}
