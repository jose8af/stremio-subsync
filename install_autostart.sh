#!/bin/zsh
# Run SubSync automatically at login (macOS LaunchAgent). Undo with: ./install_autostart.sh --remove
set -e
cd "$(dirname "$0")"
LABEL=community.subsync
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/subsync.log"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [[ "$1" == "--remove" ]]; then
  rm -f "$PLIST"
  echo "SubSync autostart removed."
  exit 0
fi

chmod +x run.sh
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$PWD/run.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
  <key>EnvironmentVariables</key><dict>
    <key>SUBSYNC_LANGUAGES</key><string>${SUBSYNC_LANGUAGES:-spa,eng}</string>
  </dict>
</dict></plist>
EOF
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "SubSync will now start at login. Logs: $LOG"
echo "Install in Stremio: http://127.0.0.1:7979/manifest.json"
