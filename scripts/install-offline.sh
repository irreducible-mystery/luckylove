#!/usr/bin/env bash
# Install Paraconsistent Annotated Evidential Logic Eτ from the vendored
# artefacts, with no network access.
#
#   ./scripts/install-offline.sh [venv-path]     (default: ./.venv)

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${1:-$repo_root/.venv}"
vendor="$repo_root/vendor/paraconsistent"

echo "==> verifying vendored artefacts"
(cd "$vendor" && sha256sum -c SHA256SUMS)

echo "==> creating virtualenv at $venv"
python3 -m venv "$venv"

echo "==> installing paraconsistent==0.2.4 offline from $vendor"
"$venv/bin/pip" install --quiet --no-index --find-links "$vendor" \
    --require-hashes -r "$repo_root/requirements.txt"

echo "==> installed"
"$venv/bin/python" -c "
from paraconsistent import ParaconsistentBlock
c = ParaconsistentBlock(mu=1.0, lam=1.0).complete
print(f'  p(1,1) -> Dc={c.dc}, Dct={c.dct}, state={c.label!r}  (inconsistent, not explosive)')
"

echo
echo "Next:"
echo "  $venv/bin/python examples/etau_demo.py"
echo "  $venv/bin/python tests/test_etau.py"
