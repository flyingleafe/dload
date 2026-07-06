#!/usr/bin/env bash
# CLI walkthrough: the same lifecycle as the Python examples (commit, list,
# inspect, pull, pin/unpin, delete, gc, clear-cache) but entirely through
# `dload` on the command line. Runs against its own "examples-cli/" prefix
# in the same R2 bucket, so it never touches the datasets committed by the
# numbered Python examples.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

set -a
. .env
set +a
export DLOAD_BUCKET=dload-test
export DLOAD_PREFIX=examples-cli/
export DLOAD_CACHE_DIR=.dload-cache-cli
export DLOAD_CACHE_BUDGET=unlimited

section() { echo; echo "=== $* ==="; }

section "creating a tiny demo directory (.cli-demo/)"
rm -rf .cli-demo
mkdir -p .cli-demo
cat >.cli-demo/hello.txt <<'EOF'
Hello from the dload CLI walkthrough.
EOF
cat >.cli-demo/hello.json <<'EOF'
{"greeting": "hello", "n": 1}
EOF
cat >.cli-demo/world.txt <<'EOF'
A second sample, paired the same way.
EOF
cat >.cli-demo/world.json <<'EOF'
{"greeting": "world", "n": 2}
EOF
ls -la .cli-demo

section "dload status"
uv run dload status

section "dload commit demo --from .cli-demo"
uv run dload commit demo --from .cli-demo --meta origin=walkthrough \
    --recipe examples/08_cli_walkthrough.sh

section "dload ls"
uv run dload ls

section "dload info demo"
uv run dload info demo

section "dload recipe demo"
uv run dload recipe demo

section "dload pull demo"
uv run dload pull demo

section "dload cache status"
uv run dload cache status

section "dload pin/unpin demo (isolated lock file)"
rm -rf .cli-lock-scratch
mkdir -p .cli-lock-scratch
(
    cd .cli-lock-scratch
    uv run dload pin demo
    echo "--- dload.lock ---"
    cat dload.lock
    uv run dload unpin demo
)
rm -rf .cli-lock-scratch

section "dload rm demo --yes"
uv run dload rm demo --yes

section "dload gc --yes"
uv run dload gc --yes

section "dload cache status (before clear)"
uv run dload cache status

section "dload cache clear --yes"
uv run dload cache clear --yes

section "cleaning up transient .cli-demo/"
rm -rf .cli-demo

echo
echo "walkthrough complete."
