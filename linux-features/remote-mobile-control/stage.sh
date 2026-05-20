#!/bin/bash
set -euo pipefail

feature_marker_dir="$INSTALL_DIR/.codex-linux"
feature_marker="$feature_marker_dir/remote-mobile-control-enabled"

mkdir -p "$feature_marker_dir"
printf '%s\n' "remote-mobile-control" > "$feature_marker"
