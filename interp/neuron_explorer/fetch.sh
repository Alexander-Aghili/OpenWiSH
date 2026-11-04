#!/bin/sh
# Fetch neuron-explorer (github.com/lyramakesmusic/neuron-explorer, MIT) at a pinned commit into interp/vendor/.
# Unmodified, ignored by git, and never imported by the harness. Its caches are large (about 230 MB).
#   sh interp/neuron_explorer/fetch.sh            then follow interp/vendor/neuron-explorer/README.md
set -eu
COMMIT=7ec363d2d9686802a1b4aa7c6d8a2627b1759159
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST="$HERE/vendor/neuron-explorer"
mkdir -p "$HERE/vendor"
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none https://github.com/lyramakesmusic/neuron-explorer.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMMIT"
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
echo "neuron-explorer at $COMMIT in $DEST"
