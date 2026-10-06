#!/usr/bin/env bash
# Put "ptq-bench" in the application menu (Linux desktops that follow the freedesktop
# standard: GNOME, KDE, XFCE...). Clicking it opens a terminal that starts `./ptq ui`
# and a browser tab with the page. Run once:  scripts/desktop-shortcut.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
target="${XDG_DATA_HOME:-$HOME/.local/share}/applications/ptq-bench.desktop"
mkdir -p "$(dirname "$target")"
cat > "$target" <<EOF
[Desktop Entry]
Type=Application
Name=ptq-bench
Comment=Measure how much a language model loses at fewer bits
Exec=$root/ptq ui
Path=$root
Terminal=true
Categories=Science;Education;
EOF
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$(dirname "$target")" || true
echo "installed $target  (look for 'ptq-bench' in the application menu)"
