#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Build the macOS app for a GitHub release.
#
# Archives the Release configuration for Apple silicon, signed with the Developer ID Application
# certificate of the team in Signing.xcconfig (hardened runtime, secure timestamp, no provisioning
# profile), has Apple notarize it through the account signed in to Xcode, and zips the stapled app
# with LICENSE, NOTICE and the license texts of everything compiled into it.
#
#   ./scripts/package-mac-app.sh
#
# NOTARIZE=0 skips notarization, for a local test of the signed app. BUNDLE_ID overrides the release
# build's bundle identifier. The zip and its SHA-256 are written to .mac-release/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${MAC_RELEASE_DIR:-$ROOT/.mac-release}"
BUNDLE_ID="${BUNDLE_ID:-wang.wangdongdong.MobileDiffuser}"

TEAM="$(sed -n 's/^DEVELOPMENT_TEAM *= *//p' "$ROOT/Signing.xcconfig" 2>/dev/null | tr -d '[:space:]')"
if [ -z "$TEAM" ]; then
    echo "error: set DEVELOPMENT_TEAM in Signing.xcconfig (see Signing.xcconfig.example)." >&2
    exit 1
fi
identities="$(security find-identity -v -p codesigning)"
if ! grep -q "Developer ID Application: .*($TEAM)" <<<"$identities"; then
    echo "error: no Developer ID Application certificate for the team in Signing.xcconfig." >&2
    exit 1
fi

# check_signature <app>: Developer ID, hardened runtime and a secure timestamp, as notarization needs.
check_signature() {
    codesign --verify --deep --strict "$1"
    local signature
    signature="$(codesign -dvv "$1" 2>&1)"
    for expected in "Authority=Developer ID Application" "Timestamp=" "(runtime)"; do
        grep -qF "$expected" <<<"$signature" || { echo "error: $1 lacks '$expected' in its signature." >&2; exit 1; }
    done
}

mkdir -p "$WORK"
ARCHIVE="$WORK/MobileDiffuser.xcarchive"
rm -rf "$ARCHIVE"
echo "Archiving the Release app …"
# Apple silicon only, like MLX and the stable-diffusion.cpp build. On macOS the app needs no
# entitlements (the entitlements file only carries iOS's increased memory limit), so it is signed
# straight with the Developer ID certificate, without a provisioning profile.
xcodebuild archive -project "$ROOT/MobileDiffuser.xcodeproj" -scheme MobileDiffuser -configuration Release \
    -destination 'generic/platform=macOS' -archivePath "$ARCHIVE" -derivedDataPath "$WORK/DerivedData" \
    -skipMacroValidation -skipPackagePluginValidation ARCHS=arm64 \
    PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" DEVELOPMENT_TEAM="$TEAM" \
    CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY="Developer ID Application" PROVISIONING_PROFILE_SPECIFIER= \
    CODE_SIGN_ENTITLEMENTS= ENABLE_HARDENED_RUNTIME=YES OTHER_CODE_SIGN_FLAGS=--timestamp \
    > "$WORK/archive.log" 2>&1 || { grep -E "error:|FAILED" "$WORK/archive.log" | tail -20 >&2; exit 1; }
APP="$ARCHIVE/Products/Applications/MobileDiffuser.app"
check_signature "$APP"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")"

if [ "${NOTARIZE:-1}" != 0 ]; then
    echo "Uploading $VERSION for notarization …"
    OPTIONS="$WORK/ExportOptions.plist"
    cat > "$OPTIONS" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key><string>developer-id</string>
    <key>destination</key><string>upload</string>
    <key>signingStyle</key><string>manual</string>
    <key>signingCertificate</key><string>Developer ID Application</string>
    <key>teamID</key><string>$TEAM</string>
</dict>
</plist>
PLIST
    xcodebuild -exportArchive -archivePath "$ARCHIVE" -exportOptionsPlist "$OPTIONS" \
        -exportPath "$WORK/upload" -allowProvisioningUpdates > "$WORK/notarize-upload.log" 2>&1 ||
        { tail -20 "$WORK/notarize-upload.log" >&2; exit 1; }

    # The notarized app can be exported once Apple has issued its ticket, usually within minutes.
    echo "Waiting for Apple's notary service …"
    NOTARIZED="$WORK/notarized"
    for attempt in $(seq 1 60); do
        rm -rf "$NOTARIZED"
        if xcodebuild -exportNotarizedApp -archivePath "$ARCHIVE" -exportPath "$NOTARIZED" \
            > "$WORK/notarize-export.log" 2>&1; then
            break
        fi
        if [ "$attempt" = 60 ]; then
            tail -20 "$WORK/notarize-export.log" >&2
            echo "error: the app was not notarized within 30 minutes." >&2
            exit 1
        fi
        sleep 30
    done
    APP="$NOTARIZED/MobileDiffuser.app"
    check_signature "$APP"
    xcrun stapler validate "$APP"
    spctl --assess --type execute "$APP"
else
    echo "NOTARIZE=0: the app is signed but not notarized."
fi

# The license texts of every package and prebuilt library compiled into the app travel with it.
NOTICES="$WORK/THIRD-PARTY-NOTICES.txt"
add_license() {   # <title> <file>
    printf '%s\n%s\n%s\n\n' "================================================================================" "$1" \
        "================================================================================"
    cat "$2"
    printf '\n\n'
}
{
    echo "MobileDiffuser for Mac is built with the following software. Their license texts follow."
    echo
    for dir in "$WORK"/DerivedData/SourcePackages/checkouts/*/ "$ROOT/../z-image-swift-mlx/"; do
        name="$(basename "$dir")"
        find "$dir" -maxdepth 4 -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \) \
            -not -path '*/.build/*' -not -path '*/Tests/*' -not -path '*/Examples/*' -not -path '*/.git/*' |
            sort | while IFS= read -r file; do add_license "$name: ${file#"$dir"}" "$file"; done
    done
    # A local Vendor/ build takes precedence over the downloaded release, as in SDCppEngine/Package.swift.
    sdcpp_licenses="$ROOT/SDCppEngine/Vendor/sdcpp.xcframework/Licenses"
    if [ ! -d "$sdcpp_licenses" ]; then
        sdcpp_licenses="$(find "$WORK/DerivedData/SourcePackages/artifacts" -type d \
            -path '*sdcpp.xcframework/Licenses' | head -1)"
    fi
    for file in "$sdcpp_licenses"/*; do
        add_license "stable-diffusion.cpp release: $(basename "$file")" "$file"
    done
} > "$NOTICES"

STAGE="$WORK/MobileDiffuser"
ZIP="$WORK/MobileDiffuser-$VERSION-macOS.zip"
rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/MobileDiffuser.app"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$NOTICES" "$STAGE/"
ditto -c -k --keepParent "$STAGE" "$ZIP"

echo
echo "Release asset: $ZIP"
echo "SHA-256: $(shasum -a 256 "$ZIP" | cut -d' ' -f1)"
