#!/usr/bin/env bash
# Download prebuilt SD3 Core ML resource folders before Xcode copies resources.
set -euo pipefail

ROOT="${SRCROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ID="${MOBILEDIFFUSER_MODEL_REPO:-Wenwu2000/MobileDiffuser-SD3-medium}"
HF_URL="https://huggingface.co/${REPO_ID}"
CACHE_ROOT="${MOBILEDIFFUSER_MODEL_CACHE:-$HOME/Library/Caches/MobileDiffuser}"
CACHE_DIR="$CACHE_ROOT/$(echo "$REPO_ID" | tr '/' '_')"

REQUIRED_DIRS=(
  "coremlsd3_2step"
  "coremlsd3_4step"
)

has_required_file() {
  local dir="$1"
  [[ -d "$ROOT/$dir" ]] && [[ -d "$ROOT/$dir/MultiModalDiffusionTransformerStage0.mlmodelc" ]]
}

all_resources_ready() {
  local dir
  for dir in "${REQUIRED_DIRS[@]}"; do
    has_required_file "$dir" || return 1
  done
}

copy_from_cache() {
  local dir
  for dir in "${REQUIRED_DIRS[@]}"; do
    [[ -d "$CACHE_DIR/$dir" ]] || return 1
  done

  echo "[MobileDiffuser] Copying SD3 resources from cache: $CACHE_DIR"
  for dir in "${REQUIRED_DIRS[@]}"; do
    rm -rf "$ROOT/$dir"
    cp -R "$CACHE_DIR/$dir" "$ROOT/$dir"
  done
}

download_with_hf_cli() {
  command -v hf >/dev/null 2>&1 || return 1

  echo "[MobileDiffuser] Downloading SD3 Core ML resources with Hugging Face CLI..."
  mkdir -p "$CACHE_DIR"
  hf download "$REPO_ID" \
    --include "coremlsd3_2step/**" "coremlsd3_4step/**" \
    --local-dir "$CACHE_DIR"
}

download_with_git_lfs() {
  command -v git >/dev/null 2>&1 || return 1
  command -v git-lfs >/dev/null 2>&1 || return 1

  echo "[MobileDiffuser] Downloading SD3 Core ML resources with git-lfs..."
  mkdir -p "$CACHE_ROOT"

  if [[ ! -d "$CACHE_DIR/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "$HF_URL" "$CACHE_DIR"
  fi

  git -C "$CACHE_DIR" fetch origin main
  git -C "$CACHE_DIR" checkout main
  git -C "$CACHE_DIR" pull --ff-only
  git -C "$CACHE_DIR" lfs install --local
  git -C "$CACHE_DIR" lfs pull \
    --include "coremlsd3_2step/**,coremlsd3_4step/**" \
    --exclude "checkpoints/**"
}

if [[ "${MOBILEDIFFUSER_SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "[MobileDiffuser] Skipping automatic model download because MOBILEDIFFUSER_SKIP_MODEL_DOWNLOAD=1."
  exit 0
fi

if all_resources_ready; then
  echo "[MobileDiffuser] SD3 resources already exist."
  exit 0
fi

mkdir -p "$CACHE_ROOT"

if ! copy_from_cache; then
  if download_with_hf_cli || download_with_git_lfs; then
    copy_from_cache
  else
    cat >&2 <<EOF
[MobileDiffuser] Missing SD3 Core ML resources and no supported downloader was found.

Install one of:
  brew install git-lfs
  pip install -U huggingface_hub

Then rebuild, or download manually:
  hf download $REPO_ID --include "coremlsd3_2step/**" "coremlsd3_4step/**" --local-dir "$ROOT"

Expected folders:
  $ROOT/coremlsd3_2step
  $ROOT/coremlsd3_4step
EOF
    exit 1
  fi
fi

if ! all_resources_ready; then
  echo "[MobileDiffuser] Download finished, but required SD3 resources are still incomplete." >&2
  exit 1
fi

echo "[MobileDiffuser] SD3 resources are ready."
