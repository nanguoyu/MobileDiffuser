#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Build the stable-diffusion.cpp XCFramework that SDCppEngine vendors.
#
# The framework is NOT committed. This script clones stable-diffusion.cpp (and its patched ggml
# submodule) at a pinned commit, builds a static library for macOS, iOS and the iOS simulator with
# the Metal backend embedded (no .metallib to ship), merges each slice's static libraries into one,
# and packages them with the public C header and a module map. Re-run after a fresh clone.
#
#   ./scripts/build-sdcpp-xcframework.sh
#
# Requirements: Xcode + CMake. Takes a few minutes.
set -euo pipefail

# master-900: the first tag carrying Qwen-Image-2.1 (#1994) together with its RGBA input fix (#2021).
SDCPP_TAG="master-900-c92d73c"
SDCPP_COMMIT="c92d73c"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/SDCppEngine/Vendor/sdcpp.xcframework"
WORK="${SDCPP_DIR:-$ROOT/.sdcpp-build}"
SRC="$WORK/stable-diffusion.cpp"
JOBS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)"

if [ ! -d "$SRC/.git" ]; then
    echo "Cloning stable-diffusion.cpp $SDCPP_TAG into $SRC …"
    mkdir -p "$WORK"
    git clone --depth 1 --branch "$SDCPP_TAG" --recurse-submodules --shallow-submodules \
        https://github.com/leejet/stable-diffusion.cpp "$SRC"
fi
actual="$(git -C "$SRC" rev-parse --short=7 HEAD)"
if [ "$actual" != "$SDCPP_COMMIT" ]; then
    echo "error: $SRC is at $actual, expected $SDCPP_COMMIT. Remove $WORK and re-run." >&2
    exit 1
fi

# Local fixes on top of the pinned commit; each patch explains itself in its header.
for patch in "$ROOT"/scripts/sdcpp-patches/*.patch; do
    if git -C "$SRC" apply --reverse --check "$patch" 2>/dev/null; then
        continue   # already applied by an earlier run
    fi
    echo "Applying $(basename "$patch") …"
    git -C "$SRC" apply "$patch"
done

COMMON_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF
    -DSD_BUILD_EXAMPLES=OFF
    -DSD_WEBP=OFF
    -DSD_WEBM=OFF
    -DSD_METAL=ON
    -DGGML_METAL_EMBED_LIBRARY=ON
    -DGGML_NATIVE=OFF
    -DGGML_OPENMP=OFF
    -DGGML_BLAS=OFF
)

# build_slice <name> <extra cmake flags…>
build_slice() {
    local name="$1"; shift
    local build="$WORK/build-$name"
    echo "Building $name …"
    cmake -S "$SRC" -B "$build" "${COMMON_FLAGS[@]}" "$@" > "$WORK/cmake-$name.log" 2>&1
    cmake --build "$build" --config Release -j "$JOBS" > "$WORK/build-$name.log" 2>&1
    # One static library per slice: the engine plus every ggml backend it was built with.
    mkdir -p "$WORK/out/$name"
    find "$build" -name '*.a' -print0 | xargs -0 libtool -static -o "$WORK/out/$name/libsdcpp.a" 2>/dev/null
    echo "  $name: $(du -h "$WORK/out/$name/libsdcpp.a" | cut -f1)"
}

build_slice macos \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
build_slice ios \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0
build_slice ios-simulator \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphonesimulator \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0

# The public API is a single self-contained C header (ggml types are only forward-declared).
HEADERS="$WORK/out/include"
rm -rf "$HEADERS"
mkdir -p "$HEADERS"
cp "$SRC/include/stable-diffusion.h" "$HEADERS/"
cat > "$HEADERS/module.modulemap" <<'MODULEMAP'
module StableDiffusionCpp {
    header "stable-diffusion.h"
    link "c++"
    link framework "Accelerate"
    link framework "Foundation"
    link framework "Metal"
    export *
}
MODULEMAP

echo "Creating XCFramework …"
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
xcodebuild -create-xcframework \
    -library "$WORK/out/macos/libsdcpp.a" -headers "$HEADERS" \
    -library "$WORK/out/ios/libsdcpp.a" -headers "$HEADERS" \
    -library "$WORK/out/ios-simulator/libsdcpp.a" -headers "$HEADERS" \
    -output "$DEST" > "$WORK/xcframework.log" 2>&1

echo "Done. $(du -sh "$DEST" | cut -f1) at $DEST (stable-diffusion.cpp $SDCPP_TAG)"
