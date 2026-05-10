"""Guard against silent test deletion.

Pillar 2 — Stability: a PR that deletes tests without justification is a
common way to silence flakes or hide regressions. This script collects
the current pytest test count and compares it to the committed baseline
in ``.github/test-count-baseline.txt``.

Behavior:
- Current count > baseline -> OK (you added tests, congrats)
- Current count == baseline -> OK (refactor without losing coverage)
- Current count < baseline -> FAIL unless ``skip-test-count`` PR label is set
- Baseline missing -> writes the current value, exits 0 (bootstrap)

Run locally:
    python scripts/check_test_count.py            # check vs baseline
    python scripts/check_test_count.py --update   # rewrite baseline (deliberate)

In CI (.github/workflows/quality-gate.yml):
    - PR_LABELS env var carries the comma-separated PR labels
    - Fails the job when count drops without the bypass label
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = REPO_ROOT / ".github" / "test-count-baseline.txt"
TESTS_DIR = REPO_ROOT / "tests"
BYPASS_LABEL = "skip-test-count"


def collect_test_count() -> int:
    """Run ``pytest --collect-only -q`` and return the integer test count."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS_DIR), "--collect-only", "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    if result.returncode != 0:
        print("[ERROR] pytest --collect-only failed; cannot count tests:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(2)

    # pytest -q prints something like:
    #     "144 tests collected in 1.23s"
    # We grab the first integer on the summary line.
    for line in reversed(result.stdout.splitlines()):
        match = re.search(r"(\d+)\s+tests?\s+collected", line)
        if match:
            return int(match.group(1))
    print("[ERROR] Could not parse test count from pytest output:", file=sys.stderr)
    print(result.stdout, file=sys.stderr)
    raise SystemExit(2)


def read_baseline() -> int | None:
    if not BASELINE_FILE.exists():
        return None
    raw = BASELINE_FILE.read_text(encoding="utf-8").strip()
    try:
        return int(raw)
    except ValueError:
        print(f"[ERROR] Baseline file is not a plain integer: {raw!r}", file=sys.stderr)
        raise SystemExit(2)


def write_baseline(count: int) -> None:
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(f"{count}\n", encoding="utf-8")


def has_bypass_label() -> bool:
    raw = os.environ.get("PR_LABELS", "")
    labels = {label.strip() for label in raw.split(",") if label.strip()}
    return BYPASS_LABEL in labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Write current count as new baseline.")
    args = parser.parse_args()

    current = collect_test_count()

    if args.update:
        write_baseline(current)
        print(f"[OK] Baseline updated to {current} tests.")
        return 0

    baseline = read_baseline()
    if baseline is None:
        write_baseline(current)
        print(f"[OK] No baseline existed; bootstrapped at {current} tests.")
        return 0

    if current >= baseline:
        delta = current - baseline
        print(f"[OK] Test count {current} >= baseline {baseline} (delta +{delta}).")
        return 0

    if has_bypass_label():
        print(f"[WARN] Test count dropped {baseline} -> {current} but PR has '{BYPASS_LABEL}' label — bypassing.")
        return 0

    print(
        f"[FAIL] Test count regressed: baseline={baseline}, current={current}.\n"
        f"       Either restore the missing tests, or apply the '{BYPASS_LABEL}' PR label "
        f"with a written justification in the PR description.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
