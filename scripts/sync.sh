#!/usr/bin/env sh
# Sync the project environment, then repair the macOS + iCloud damage.
#
# `uv` marks the virtualenv hidden (UF_HIDDEN). Inside an iCloud-synced tree
# (~/Documents by default on macOS) that flag is propagated to every file
# created in it, and this Python build skips hidden .pth files — so the
# editable install silently stops working:
#
#     ModuleNotFoundError: No module named 'myriapod'
#
# It comes back after every sync, hence this wrapper. The durable fix is to
# keep the project (or at least its venv) out of the synced tree; see the
# macOS + iCloud section of CLAUDE.md.
#
# Usage: ./scripts/sync.sh [extra uv sync args]
set -eu

VENV="${UV_PROJECT_ENVIRONMENT:-.venv}"

uv sync --all-extras "$@"

if [ "$(uname)" = "Darwin" ] && command -v chflags >/dev/null 2>&1; then
    chflags -R nohidden "$VENV" 2>/dev/null || true
fi

# Prove the editable install actually resolves, rather than trusting it.
if uv run python -c "import myriapod" 2>/dev/null; then
    echo "environment ready: $VENV"
else
    echo "environment is still broken: 'import myriapod' fails from $VENV" >&2
    echo "diagnose with: $VENV/bin/python -v -c pass | grep pth" >&2
    exit 1
fi
