// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import CoreGraphics
import Foundation
import StableDiffusionCpp

/// Conversions between sd.cpp's `sd_image_t` (tightly packed, 8-bit, 3 or 4 interleaved channels,
/// straight alpha) and `CGImage`.
enum SDImage {

    /// Copies the pixels out, so the caller can free the sd.cpp buffer immediately afterwards.
    static func makeCGImage(_ image: sd_image_t) -> CGImage? {
        let width = Int(image.width), height = Int(image.height), channels = Int(image.channel)
        guard width > 0, height > 0, channels == 3 || channels == 4, let pixels = image.data else { return nil }
        let bytesPerRow = width * channels
        let data = Data(bytes: pixels, count: bytesPerRow * height)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        // sd.cpp writes straight (unassociated) alpha, which is what transparent PNG export wants.
        let alpha: CGImageAlphaInfo = channels == 4 ? .last : .none
        return CGImage(width: width, height: height, bitsPerComponent: 8,
                       bitsPerPixel: 8 * channels, bytesPerRow: bytesPerRow,
                       space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)
    }

    /// Frees every buffer `generate_image` returned: each image's pixels, then the array itself.
    static func free(_ images: UnsafeMutablePointer<sd_image_t>?, count: Int) {
        guard let images else { return }
        for i in 0..<max(0, count) { Foundation.free(images[i].data) }
        Foundation.free(images)
    }
}

/// Pixels for an image passed INTO sd.cpp (a reference image for editing). Owns its buffer for the
/// duration of the call; `sd_image_t` only borrows it.
final class SDInputImage {
    let width: Int
    let height: Int
    let channels: Int
    private let pixels: UnsafeMutablePointer<UInt8>

    /// Keeps an alpha channel when the source has one: Qwen-Image-2.1 edits transparent layers, and
    /// flattening them here would silently discard the transparency the user asked to edit.
    init?(_ image: CGImage) {
        let width = image.width, height = image.height
        let hasAlpha: Bool
        switch image.alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast: hasAlpha = false
        default: hasAlpha = true
        }
        let channels = hasAlpha ? 4 : 3
        guard width > 0, height > 0 else { return nil }

        // Draw into RGBA first (CoreGraphics has no packed 24-bit destination), un-premultiply, and
        // then drop alpha if the source had none.
        let rgbaRow = width * 4
        var rgba = [UInt8](repeating: 0, count: rgbaRow * height)
        let drawn: Bool = rgba.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: rgbaRow,
                                          space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }

        let pixels = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height * channels)
        for p in 0..<(width * height) {
            let a = Int(rgba[p * 4 + 3])
            for c in 0..<3 {
                let v = Int(rgba[p * 4 + c])
                pixels[p * channels + c] = hasAlpha && a > 0 && a < 255 ? UInt8(min(255, v * 255 / a)) : UInt8(v)
            }
            if hasAlpha { pixels[p * channels + 3] = UInt8(a) }
        }
        self.width = width
        self.height = height
        self.channels = channels
        self.pixels = pixels
    }

    deinit { pixels.deallocate() }

    var raw: sd_image_t {
        sd_image_t(width: UInt32(width), height: UInt32(height), channel: UInt32(channels), data: pixels)
    }
}
