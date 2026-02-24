#!/usr/bin/env bash
# Re-download the DINOv2 base model into checkpoints/dinov2-base.
# Use this if you see: safetensors_rust.SafetensorError: Error while deserializing header: header too large
# (usually because model.safetensors was truncated or is a stub.)

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${1:-checkpoints/dinov2-base}"
HF_MODEL="${2:-facebook/dinov2-base}"

if [[ -d "$BASE_DIR" ]]; then
  echo "Removing existing $BASE_DIR (current model.safetensors may be corrupted or stub)."
  rm -rf "$BASE_DIR"
fi

echo "Downloading $HF_MODEL into $BASE_DIR"
# Note: install_huggingface_model.sh maps -d to model path, -m to install dir (opposite of usage text)
exec "$REPO_ROOT/scripts/install_huggingface_model.sh" -d "$HF_MODEL" -m "$BASE_DIR"
