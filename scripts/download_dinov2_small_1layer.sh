#!/usr/bin/env bash
# Download facebook/dinov2-small-imagenet1k-1-layer into checkpoints/dinov2-small-imagenet1k-1-layer.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${1:-checkpoints/dinov2-small-imagenet1k-1-layer}"
HF_MODEL="${2:-facebook/dinov2-small-imagenet1k-1-layer}"

if [[ -d "$BASE_DIR" ]]; then
  echo "Removing existing $BASE_DIR."
  rm -rf "$BASE_DIR"
fi

echo "Downloading $HF_MODEL into $BASE_DIR"
exec "$REPO_ROOT/scripts/install_huggingface_model.sh" -d "$HF_MODEL" -m "$BASE_DIR"
