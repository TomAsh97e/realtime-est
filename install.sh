#!/usr/bin/env bash
# Manual install (alternative to the plugin marketplace):
#   copies the skill to ~/.claude/skills/realtime-est and registers the Stop hook
#   in ~/.claude/settings.json (a backup is written next to it).
#   ./install.sh            install / update
#   ./install.sh --uninstall remove skill + hook (your recorded data is kept)
set -euo pipefail

CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
SKILL_DST="$CLAUDE_DIR/skills/realtime-est"
SETTINGS="$CLAUDE_DIR/settings.json"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/skills/realtime-est"
HOOK_CMD="python3 \"$SKILL_DST/scripts/rte.py\" hook"

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }

edit_settings() { # $1 = add|remove
  python3 - "$SETTINGS" "$HOOK_CMD" "$1" <<'PY'
import json, os, shutil, sys
path, cmd, mode = sys.argv[1:4]
data = {}
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
    shutil.copy(path, path + ".bak-realtime-est")
hooks = data.setdefault("hooks", {})
stop = hooks.get("Stop", [])
# drop any previous realtime-est hook
for group in stop:
    group["hooks"] = [h for h in group.get("hooks", []) if "rte.py" not in h.get("command", "") or "realtime-est" not in h.get("command", "")]
stop = [g for g in stop if g.get("hooks")]
if mode == "add":
    stop.append({"hooks": [{"type": "command", "command": cmd, "timeout": 15}]})
if stop:
    hooks["Stop"] = stop
else:
    hooks.pop("Stop", None)
if not hooks:
    data.pop("hooks", None)
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
}

if [[ "${1:-}" == "--uninstall" ]]; then
  rm -rf "$SKILL_DST"
  edit_settings remove
  echo "realtime-est removed. Recorded data kept in $CLAUDE_DIR/realtime-est (delete it manually if you want)."
  exit 0
fi

mkdir -p "$CLAUDE_DIR/skills"
rm -rf "$SKILL_DST"
cp -r "$SRC" "$SKILL_DST"
rm -rf "$SKILL_DST/scripts/__pycache__"
edit_settings add
echo "Installed skill to $SKILL_DST and registered the Stop hook in $SETTINGS"
echo "Importing your existing Claude Code history..."
python3 "$SKILL_DST/scripts/rte.py" backfill
python3 "$SKILL_DST/scripts/rte.py" profile --no-backfill || true
echo
echo "Done. Restart Claude Code (or open a new session) to load the skill and hook."
