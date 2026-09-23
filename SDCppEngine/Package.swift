// swift-tools-version: 5.10
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The second inference engine: stable-diffusion.cpp (ggml, Metal) behind the same
// `DiffusionEngine` protocol as the MLX engines. It runs GGUF models, which MLX cannot read,
// and brings sd.cpp's own model implementations with it, so a model it supports needs no port.
//
// stable-diffusion.cpp comes as a prebuilt XCFramework published in this repository's releases.
// `scripts/build-sdcpp-xcframework.sh` builds it; a framework it built locally into Vendor/ takes
// precedence over the download, so a new sd.cpp version or patch can be tried before it is released.

import Foundation
import PackageDescription

let localFramework = "Vendor/sdcpp.xcframework"
let stableDiffusionCpp: Target =
    FileManager.default.fileExists(atPath: Context.packageDirectory + "/" + localFramework)
    ? .binaryTarget(name: "StableDiffusionCpp", path: localFramework)
    : .binaryTarget(
        name: "StableDiffusionCpp",
        url: "https://github.com/nanguoyu/MobileDiffuser/releases/download/sdcpp-2a4ebba.1/sdcpp.xcframework.zip",
        checksum: "9a5821d6691f516034505d65e6d09149ef856b385a9efe9db1792e4e46e7161a")

let package = Package(
    name: "SDCppEngine",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "SDCppEngine", targets: ["SDCppEngine"])],
    dependencies: [
        .package(url: "https://github.com/nanguoyu/swift-diffusion-core", branch: "main"),
    ],
    targets: [
        stableDiffusionCpp,
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
