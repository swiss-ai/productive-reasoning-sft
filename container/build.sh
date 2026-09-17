#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/synthetic-sft-v0.1.sqsh" >&2
  exit 2
fi

target=$1
if [[ $target != /* ]]; then
  echo "target must be an absolute path" >&2
  exit 2
fi

project_dir=$(cd "$(dirname "$0")/.." && pwd)
image_name=synthetic-sft:v0.1
temporary="${target}.tmp.$$"

podman build --format docker --platform linux/arm64 -t "$image_name" \
  -f "$project_dir/container/Containerfile" "$project_dir"
mkdir -p "$(dirname "$target")"

# Some CSCS enroot versions return a non-zero status when unmounting the
# temporary Podman mount even though mksquashfs completed successfully. Build
# to a temporary path, validate the filesystem, and only then publish it.
import_status=0
enroot import -x mount -o "$temporary" "podman://$image_name" || import_status=$?
if ! unsquashfs -s "$temporary" >/dev/null 2>&1; then
  echo "enroot import failed with status $import_status and produced no valid image" >&2
  [[ $import_status -ne 0 ]] || import_status=1
  exit "$import_status"
fi
if [[ $import_status -ne 0 ]]; then
  echo "warning: enroot returned status $import_status after producing a valid SquashFS image" >&2
fi
mv -f "$temporary" "$target"
echo "$target"
