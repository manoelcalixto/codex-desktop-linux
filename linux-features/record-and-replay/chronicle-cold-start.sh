#!/usr/bin/env bash
set -Eeuo pipefail

app_dir="${1:-${CODEX_LINUX_APP_DIR:-}}"

log() {
    echo "Record & Replay Chronicle cold-start: $*"
}

truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON|enabled|ENABLED) return 0 ;;
        *) return 1 ;;
    esac
}

falsy() {
    case "${1:-}" in
        0|false|FALSE|no|NO|off|OFF|disabled|DISABLED) return 0 ;;
        *) return 1 ;;
    esac
}

chronicle_enabled_in_config() {
    local config_path="${CODEX_HOME:-$HOME/.codex}/config.toml"
    [ -f "$config_path" ] || return 1

    python3 - "$config_path" <<'PY'
import re
import sys

config_path = sys.argv[1]
try:
    raw = open(config_path, "rb").read()
except OSError:
    sys.exit(1)

text = raw.decode("utf-8", "ignore")
try:
    import tomllib

    parsed = tomllib.loads(text)
    sys.exit(0 if parsed.get("features", {}).get("chronicle") is True else 1)
except Exception:
    pass

in_features = False
for raw_line in text.splitlines():
    line = raw_line.split("#", 1)[0].strip()
    if not line:
        continue
    if line.startswith("[") and line.endswith("]"):
        in_features = line == "[features]"
        continue
    if in_features:
        match = re.match(r"chronicle\s*=\s*(true|false)\b", line, re.IGNORECASE)
        if match:
            sys.exit(0 if match.group(1).lower() == "true" else 1)
    match = re.match(r"features\s*=\s*\{(?P<body>[^}]*)\}", line, re.IGNORECASE)
    if match:
        for part in match.group("body").split(","):
            key, _, value = part.partition("=")
            if key.strip() == "chronicle":
                sys.exit(0 if value.strip().lower().startswith("true") else 1)

sys.exit(1)
PY
}

chronicle_autostart_enabled() {
    if [ -n "${CODEX_RECORD_REPLAY_CHRONICLE_AUTOSTART:-}" ]; then
        truthy "$CODEX_RECORD_REPLAY_CHRONICLE_AUTOSTART"
        return $?
    fi

    chronicle_enabled_in_config
}

find_record_replay_binary() {
    local candidate
    for candidate in \
        "${CODEX_RECORD_REPLAY_LINUX_BIN:-}" \
        "$app_dir/resources/native/codex-record-replay-linux" \
        "$app_dir/resources/plugins/openai-bundled/plugins/record-and-replay/bin/codex-record-replay-linux"; do
        [ -n "$candidate" ] || continue
        [ -x "$candidate" ] || continue
        printf '%s\n' "$candidate"
        return 0
    done

    if command -v codex-record-replay-linux >/dev/null 2>&1; then
        command -v codex-record-replay-linux
        return 0
    fi

    return 1
}

skysight_runtime_dir() {
    if [ -n "${CODEX_SKYSIGHT_RUNTIME_DIR:-}" ]; then
        printf '%s\n' "$CODEX_SKYSIGHT_RUNTIME_DIR"
        return 0
    fi
    if [ -n "${XDG_RUNTIME_DIR:-}" ]; then
        printf '%s\n' "$XDG_RUNTIME_DIR/skysight"
        return 0
    fi
    printf '%s\n' "${TMPDIR:-/tmp}/skysight"
}

start_skysight() {
    "$record_replay_bin" skysight start --interval-seconds "$interval_seconds" --summary-agent enabled >/dev/null
}

start_skysight_with_lock() {
    local runtime_dir
    local lock_path

    runtime_dir="$(skysight_runtime_dir)"
    if ! mkdir -p "$runtime_dir"; then
        log "could not create Skysight runtime dir for autostart lock: $runtime_dir"
        return 0
    fi
    chmod 700 "$runtime_dir" 2>/dev/null || true
    lock_path="$runtime_dir/autostart.lock"

    if ! command -v flock >/dev/null 2>&1; then
        log "flock not found; starting Skysight without autostart serialization"
        start_skysight
        return 0
    fi

    (
        flock 9
        start_skysight
    ) 9>"$lock_path"
}

if falsy "${CODEX_RECORD_REPLAY_CHRONICLE_AUTOSTART:-}"; then
    log "disabled by CODEX_RECORD_REPLAY_CHRONICLE_AUTOSTART"
    exit 0
fi

if ! chronicle_autostart_enabled; then
    log "Chronicle feature is not enabled in config; skipping"
    exit 0
fi

record_replay_bin="$(find_record_replay_binary || true)"
if [ -z "$record_replay_bin" ]; then
    log "codex-record-replay-linux binary not found; skipping"
    exit 0
fi

interval_seconds="${CODEX_RECORD_REPLAY_CHRONICLE_INTERVAL_SECONDS:-60}"
case "$interval_seconds" in
    ''|*[!0-9]*) interval_seconds="60" ;;
esac
if [ "$interval_seconds" -lt 1 ]; then
    interval_seconds="60"
fi

log "starting Skysight with interval ${interval_seconds}s"
start_skysight_with_lock
log "Skysight start requested"
