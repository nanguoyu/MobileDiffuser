import SwiftUI

/// Placeholder root view.
///
/// The original CoreML Stable Diffusion 3 UI and logic have been removed. The new universal
/// (macOS + iOS) MLX experience — Models / Create / Library / Settings — is being built on
/// `swift-diffusion-core`; see `docs/BLUEPRINT.md`.
struct ContentView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "sparkles")
                .font(.system(size: 44, weight: .light))
                .foregroundStyle(.secondary)
            Text("Diffuser")
                .font(.title2.weight(.medium))
            Text("Rebuilding on MLX — see docs/BLUEPRINT.md")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
