// swift-tools-version: 5.10
import PackageDescription

// AppEngines — a thin wrapper that lets the universal app pull in the macOS-only FLUX engine
// WITHOUT it entering the iOS package graph. `Flux2DiffusionEngine` is depended on only on macOS
// via `.when(platforms: [.macOS])`; on iOS this builds to an empty module. Z-Image is depended on
// directly by the app (it is cross-platform). See docs/BLUEPRINT.md ("FLUX on iOS").
//
// Why a wrapper at all: Xcode's per-build-file `platformFilter` only affects linking, not package
// resolution — SPM validates the whole graph per platform, so a macOS-only package referenced by
// the app breaks the iOS build. The SPM-native platform-conditional product dependency is only
// expressible in a Package manifest, hence this package.
let package = Package(
    name: "AppEngines",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "AppEngines", targets: ["AppEngines"])],
    dependencies: [
        .package(path: "../../flux2-diffusion-engine"),
    ],
    targets: [
        .target(
            name: "AppEngines",
            dependencies: [
                .product(name: "Flux2DiffusionEngine", package: "flux2-diffusion-engine",
                         condition: .when(platforms: [.macOS])),
            ]
        ),
    ]
)
