// swift-tools-version: 5.10
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The second inference engine: stable-diffusion.cpp (ggml, Metal) behind the same
// `DiffusionEngine` protocol as the MLX engines. It runs GGUF models, which MLX cannot read,
// and brings sd.cpp's own model implementations with it, so a model it supports needs no port.
//
// The XCFramework is not committed; build it with `scripts/build-sdcpp-xcframework.sh`.

import PackageDescription

let package = Package(
    name: "SDCppEngine",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "SDCppEngine", targets: ["SDCppEngine"])],
    dependencies: [
        .package(url: "https://github.com/nanguoyu/swift-diffusion-core", branch: "main"),
    ],
    targets: [
        .binaryTarget(name: "StableDiffusionCpp", path: "Vendor/sdcpp.xcframework"),
        .target(
            name: "SDCppEngine",
            dependencies: [
                "StableDiffusionCpp",
                .product(name: "DiffusionCore", package: "swift-diffusion-core"),
            ]
        ),
        .testTarget(
            name: "SDCppEngineTests",
            dependencies: ["SDCppEngine"]
        ),
    ]
)
