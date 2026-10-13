#!/usr/bin/env bash
# Install actual Arial from the original Microsoft core-font distribution.
# The package is extracted as an archive; the Windows executable is never run.
set -euo pipefail

for report_tool in curl cabextract sha256sum fc-cache; do
    command -v "$report_tool" >/dev/null || {
        echo "Missing $report_tool. Install curl, cabextract and fontconfig first." >&2
        exit 1
    }
done

report_font_dir="${REPORT_FONT_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/fonts/msttcorefonts}"
report_font_tmp="$(mktemp -d)"
trap 'rm -rf "$report_font_tmp"' EXIT
curl --fail --location --retry 2 --connect-timeout 15 --max-time 120 \
    'https://downloads.sourceforge.net/project/corefonts/the%20fonts/final/arial32.exe' \
    --output "$report_font_tmp/arial32.exe"
echo '85297a4d146e9c87ac6f74822734bdee5f4b2a722d7eaa584b7f2cbf76f478f6  arial32.exe' \
    > "$report_font_tmp/expected.sha256"
(cd "$report_font_tmp" && sha256sum --check expected.sha256)
mkdir -p "$report_font_dir"
cabextract -q -F '*.TTF' -d "$report_font_dir" "$report_font_tmp/arial32.exe"
fc-cache -f "$report_font_dir"
fc-match Arial
