#!/usr/bin/env bash
# Apply repo patches to the verl package inside the training venv (idempotent).
# Each patch under patches/ is a unified diff against the pristine verl 0.5.0
# wheel as installed in .venv (paths relative to site-packages/).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SITE_PKG="$PROJECT_DIR/.venv/lib/python3.12/site-packages"

shopt -s nullglob
PATCHES=("$PROJECT_DIR"/patches/*.patch)
[[ ${#PATCHES[@]} -gt 0 ]] || { echo "[apply_patches] no patches found"; exit 0; }

for p in "${PATCHES[@]}"; do
  name="$(basename "$p")"
  if patch -p1 -d "$SITE_PKG" --dry-run --silent -i "$p" >/dev/null 2>&1; then
    patch -p1 -d "$SITE_PKG" --silent -i "$p"
    echo "[apply_patches] applied: $name"
  elif patch -p1 -d "$SITE_PKG" --dry-run --forward --silent -i "$p" >/dev/null 2>&1 \
    || ! patch -p1 -d "$SITE_PKG" --dry-run --reverse --silent -i "$p" >/dev/null 2>&1; then
    # reverse-applyable => already applied
    if patch -p1 -d "$SITE_PKG" --dry-run --reverse --silent -i "$p" >/dev/null 2>&1; then
      echo "[apply_patches] already applied: $name"
    else
      echo "[apply_patches] FAILED (context mismatch): $name" >&2
      exit 1
    fi
  else
    echo "[apply_patches] already applied: $name"
  fi
done
