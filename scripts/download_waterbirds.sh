#!/usr/bin/env bash
# Download and extract the official Waterbirds release (about 490 MB).
# URL from the Group DRO repository README (github.com/kohpangwei/group_DRO).
# Usage: bash scripts/download_waterbirds.sh [destination_dir]
set -euo pipefail
DEST="${1:-/content/data}"
URL="https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
NAME="waterbird_complete95_forest2water2"
mkdir -p "$DEST"
cd "$DEST"
if [ -f "$NAME/metadata.csv" ]; then
  echo "already extracted: $DEST/$NAME"
  exit 0
fi
if [ ! -f "$NAME.tar.gz" ]; then
  wget -c --tries=3 "$URL" -O "$NAME.tar.gz.part"
  mv "$NAME.tar.gz.part" "$NAME.tar.gz"
fi
tar -xzf "$NAME.tar.gz"
sha256sum "$NAME.tar.gz" | tee "$NAME.tar.gz.sha256"
echo "extracted to $DEST/$NAME ($(find "$NAME" -name '*.jpg' | wc -l) jpg files)"
