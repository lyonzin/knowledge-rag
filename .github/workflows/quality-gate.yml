name: Quality Gate

# 7 Pillars Quality Gate — wave 2 of 5 (Security + Versioning).
# Each job below maps to a status check that branch protection requires.
# More pillars (Stability, Memory, Versatility, Scalability, Quality) are
# added by subsequent waves.

on:
  pull_request:
    branches: [master]
    # labeled/unlabeled so bypass labels (skip-changelog, skip-test-count,
    # skip-perf-gate) re-evaluate the gate without needing a forced push.
    types: [opened, synchronize, reopened, labeled, unlabeled]
  push:
    branches: [master]

permissions:
  contents: read

# Cancel earlier runs of the same PR when a new push lands
concurrency:
  group: quality-gate-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # ════════════════════════════════════════════════════════════════════════
  # PILLAR 1: SECURITY
  # ════════════════════════════════════════════════════════════════════════

  bandit:
    name: "Pillar 1 — Bandit SAST"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install bandit
      - name: Run bandit
        shell: bash
        run: |
          # Fail on HIGH severity, report MEDIUM as warning
          bandit -r mcp_server/ scripts/ --severity-level high --confidence-level medium

  semgrep:
    name: "Pillar 1 — Semgrep"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install semgrep
      - name: Run semgrep with auto config
        shell: bash
        run: |
          # --error makes ERROR-level findings fail the job; warnings are reported but pass.
          semgrep --config=auto --error mcp_server/ scripts/

  pip-audit:
    name: "Pillar 1 — pip-audit (CVEs)"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install pip-audit
      - name: Audit dependency tree (without installing knowledge-rag)
        shell: bash
        # We audit the dependency tree only — NOT the knowledge-rag package
        # itself. Auditing knowledge-rag would always fail on release PRs
        # (the version about to ship is not yet on PyPI), and we already
        # fully exercise the project via the test suite. Resolving deps via
        # pip download lets pip-audit see the real graph without installing
        # an editable / not-on-PyPI version.
        run: |
          python -m pip install --upgrade pip
          # Materialize the actual dependency closure into a requirements
          # file by asking pip to resolve (without installing) what
          # `pip install .` would pull. Then audit that file.
          pip install --upgrade pip-tools
          pip-compile --quiet --output-file=resolved-deps.txt --strip-extras pyproject.toml
          # Drop the project itself from the file (pip-compile may include it)
          grep -v "^knowledge-rag" resolved-deps.txt > resolved-deps.clean.txt || cp resolved-deps.txt resolved-deps.clean.txt

          # Strict mode fails on any unfixed advisory. Documented exceptions:
          #   CVE-2026-3219 (pip): tarball symlink TOCTOU — no upstream fix
          #     as of 2026-05. Does not affect us; users run pip themselves.
          #   CVE-2026-45829 (chromadb 1.5.9): no fix available as of 2026-06.
          #     Upstream issue; 1.5.9 is latest. Remove when chromadb patches.
          ignore_args="--ignore-vuln CVE-2026-3219 --ignore-vuln CVE-2026-45829"
          if [ -n "${IGNORE_AUDIT_IDS:-}" ]; then
            for id in ${IGNORE_AUDIT_IDS}; do
              ignore_args="${ignore_args} --ignore-vuln ${id}"
            done
          fi
          pip-audit --strict --requirement resolved-deps.clean.txt ${ignore_args}

  gitleaks:
    name: "Pillar 1 — Gitleaks (secrets)"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0  # gitleaks scans full history
      - name: Scan for secrets
        uses: gitleaks/gitleaks-action@v3
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_CONFIG: .github/gitleaks.toml

  dependency-review:
    name: "Pillar 1 — Dependency Review"
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    # Requires GitHub Dependency Graph to be enabled on the repo
    # (Settings -> Code security -> Dependency graph). Until the maintainer
    # flips that on, we run the action but do not block the gate.
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - uses: actions/dependency-review-action@v5
        with:
          fail-on-severity: high
          comment-summary-in-pr: on-failure

  # ════════════════════════════════════════════════════════════════════════
  # PILLAR 6: VERSIONING
  # ════════════════════════════════════════════════════════════════════════

  version-sync:
    name: "Pillar 6 — Version sync (3 files)"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Verify versions agree
        shell: bash
        run: python scripts/check_version_sync.py

  api-surface:
    name: "Pillar 6 — Public API surface"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Compare API surface against baseline
        shell: bash
        run: python scripts/check_api_surface.py --check

  conventional-commit-title:
    name: "Pillar 6 — Conventional commit (PR title)"
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: amannn/action-semantic-pull-request@v6
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          types: |
            feat
            fix
            docs
            chore
            refactor
            test
            perf
            build
            ci
            style
            revert
          requireScope: false
          subjectPattern: ^[A-Za-z].+
          subjectPatternError: |
            The PR title subject must start with a letter and not be empty.
            Example: feat(server): add new search filter

  changelog:
    name: "Pillar 6 — CHANGELOG entry"
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Verify ## Unreleased gained an entry (if user-facing PR)
        shell: bash
        env:
          PR_TITLE: ${{ github.event.pull_request.title }}
          PR_LABELS: ${{ join(github.event.pull_request.labels.*.name, ',') }}
        run: python scripts/check_changelog.py --base-ref origin/master

  backwards-compat:
    name: "Pillar 6 — Backwards-compat tests"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install dependencies
        shell: bash
        run: |
          pip install -e .
          pip install pytest pytest-cov pyyaml
      - name: Run backwards-compat regression tests
        shell: bash
        env:
          HF_HUB_OFFLINE: "1"
          TRANSFORMERS_OFFLINE: "1"
          HF_HUB_DISABLE_PROGRESS_BARS: "1"
        run: pytest tests/test_backwards_compat.py -v

  # ════════════════════════════════════════════════════════════════════════
  # PILLAR 4: VERSATILITY
  # ════════════════════════════════════════════════════════════════════════

  versatility-presets:
    name: "Pillar 4 — Preset matrix"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install minimal deps for YAML parsing
        shell: bash
        run: pip install pytest pyyaml
      - name: Run preset structural validation
        shell: bash
        run: pytest tests/test_presets.py -v

  versatility-locale:
    name: "Pillar 4 — Locale matrix"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install package
        shell: bash
        run: |
          pip install -e .
          pip install pytest
      - name: Run locale + encoding tests
        shell: bash
        env:
          HF_HUB_OFFLINE: "1"
          TRANSFORMERS_OFFLINE: "1"
          HF_HUB_DISABLE_PROGRESS_BARS: "1"
        run: pytest tests/test_locale.py -v

  versatility-formats:
    name: "Pillar 4 — Format smoke matrix"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install package
        shell: bash
        run: |
          pip install -e .
          pip install pytest
      - name: Run format smoke tests
        shell: bash
        env:
          HF_HUB_OFFLINE: "1"
          TRANSFORMERS_OFFLINE: "1"
          HF_HUB_DISABLE_PROGRESS_BARS: "1"
        run: pytest tests/test_format_smoke.py -v

  # ════════════════════════════════════════════════════════════════════════
  # PILLAR 7: QUALITY
  # ════════════════════════════════════════════════════════════════════════

  mypy:
    name: "Pillar 7 — mypy strict (gradual)"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install mypy + package
        shell: bash
        run: |
          pip install mypy
          pip install -e .
      - name: Run mypy on annotated modules + scripts
        shell: bash
        # Gradual rollout: only modules already fully annotated. As legacy
        # code earns annotations, append paths here.
        run: mypy mcp_server/instance_lock.py mcp_server/preflight.py scripts/

  interrogate:
    name: "Pillar 7 — Docstring coverage"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install interrogate
      - name: Verify docstring coverage >= 80%
        shell: bash
        run: interrogate -c pyproject.toml mcp_server/

  radon:
    name: "Pillar 7 — Cyclomatic complexity"
    runs-on: ubuntu-latest
    # Report-only for now: we know Config / orchestrator.query are D/E
    # (legacy hot-spots). Visibility helps the maintainer prioritize
    # refactors without blocking unrelated PRs.
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install radon
      - name: Report C+ complexity blocks
        shell: bash
        run: |
          echo "## Radon Cyclomatic Complexity (C and higher)" >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
          radon cc mcp_server/ -a -nc | tee -a $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
          # Fail only when a NEW block is rated D+ — placeholder for now.
          # The followup PR will diff against master baseline.

  vulture:
    name: "Pillar 7 — Dead code (vulture)"
    runs-on: ubuntu-latest
    continue-on-error: true   # report-only initially
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install vulture
      - name: Scan for dead code (high-confidence only)
        shell: bash
        run: |
          vulture mcp_server/ scripts/ tests/_vulture_whitelist.py --min-confidence 80 \
            | tee vulture-report.txt || true
          if [ -s vulture-report.txt ] && grep -v "^$" vulture-report.txt > /dev/null; then
            echo "## Vulture findings (informational)" >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
            cat vulture-report.txt >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
          fi

  pr-size:
    name: "Pillar 7 — PR size guard"
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    continue-on-error: true   # warn-only by design
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0
      - name: Measure code change size (excluding docs/lockfiles)
        shell: bash
        run: |
          base="${{ github.event.pull_request.base.sha }}"
          head="${{ github.event.pull_request.head.sha }}"
          changed=$(git diff --numstat "${base}" "${head}" \
            -- 'mcp_server/**' 'scripts/**' 'tests/**' 'bench/**' \
            ':!*.md' ':!*.json' ':!*.lock' ':!*.txt' ':!*.yaml' ':!*.yml' \
            | awk '{added+=$1; removed+=$2} END {print added+removed}')
          changed=${changed:-0}
          echo "## PR size: ${changed} lines changed (code only)" >> $GITHUB_STEP_SUMMARY
          if [ "${changed}" -gt 500 ]; then
            echo "" >> $GITHUB_STEP_SUMMARY
            echo "⚠️ This PR exceeds 500 lines of code change. Consider splitting into smaller PRs for easier review." >> $GITHUB_STEP_SUMMARY
            echo "" >> $GITHUB_STEP_SUMMARY
            echo "(Warning only — this does not block merge.)" >> $GITHUB_STEP_SUMMARY
          else
            echo "" >> $GITHUB_STEP_SUMMARY
            echo "✅ PR size is reasonable." >> $GITHUB_STEP_SUMMARY
          fi

  # ════════════════════════════════════════════════════════════════════════
  # PILLAR 2: STABILITY
  # ════════════════════════════════════════════════════════════════════════

  test-count:
    name: "Pillar 2 — Test count anti-regression"
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install package + pytest + collect-time deps
        shell: bash
        # The collect-only run imports every test module — including those
        # that depend on hypothesis, psutil, numpy, yaml. Install all of
        # them so collection succeeds across the full suite.
        run: |
          pip install -e .
          pip install pytest hypothesis psutil numpy pyyaml
      - name: Verify test count has not regressed
        shell: bash
        env:
          PR_LABELS: ${{ join(github.event.pull_request.labels.*.name, ',') }}
          HF_HUB_OFFLINE: "1"
          TRANSFORMERS_OFFLINE: "1"
          HF_HUB_DISABLE_PROGRESS_BARS: "1"
        run: python scripts/check_test_count.py

  # ════════════════════════════════════════════════════════════════════════
  # PILLAR 3: MEMORY
  # ════════════════════════════════════════════════════════════════════════

  memory-baseline:
    name: "Pillar 3 — Memory baseline"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install package + memory deps
        shell: bash
        run: |
          pip install -e .
          pip install pytest psutil numpy
      - name: Run memory baseline tests
        shell: bash
        env:
          HF_HUB_OFFLINE: "1"
          TRANSFORMERS_OFFLINE: "1"
          HF_HUB_DISABLE_PROGRESS_BARS: "1"
        run: pytest tests/test_memory_baseline.py -v

  property-fuzz:
    name: "Pillar 4 (innovation) — Property-based fuzz"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install package + hypothesis
        shell: bash
        run: |
          pip install -e .
          pip install pytest hypothesis
      - name: Run property-based parser tests
        shell: bash
        env:
          HF_HUB_OFFLINE: "1"
          TRANSFORMERS_OFFLINE: "1"
          HF_HUB_DISABLE_PROGRESS_BARS: "1"
        run: pytest tests/test_ingestion_property.py -v

  # ════════════════════════════════════════════════════════════════════════
  # PILLAR 5: SCALABILITY (perf gate)
  # ════════════════════════════════════════════════════════════════════════

  perf-bench:
    name: "Pillar 5 — Performance regression gate (10%)"
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    # Promoted to required at v3.9.0. Bootstrap-resilient: when master
    # does not yet contain bench/, the gate skips with [INFO] message
    # rather than failing.
    steps:
      - name: Checkout PR branch
        uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install package + bench deps
        shell: bash
        run: |
          pip install -e .
          pip install pytest pytest-benchmark psutil numpy
      - name: Bench against PR branch
        shell: bash
        env:
          HF_HUB_OFFLINE: "1"
          TRANSFORMERS_OFFLINE: "1"
          HF_HUB_DISABLE_PROGRESS_BARS: "1"
        run: pytest bench/ --benchmark-only --benchmark-min-rounds=3 --benchmark-json=branch.json
      - name: Checkout master and bench it
        shell: bash
        # First-run resilience: if master does not yet contain bench/ (the
        # PR that introduces it), the perf gate is informational. Once bench/
        # exists in master, the gate compares medians and enforces the 10%
        # ceiling.
        run: |
          git fetch origin master:master-base
          if git show master-base:bench/test_bench_search.py > /dev/null 2>&1; then
            git checkout master-base -- bench/ scripts/ mcp_server/
            pytest bench/ --benchmark-only --benchmark-min-rounds=3 --benchmark-json=master.json || \
              echo "[WARN] master bench step failed; perf gate becomes informational this run"
            git checkout HEAD -- bench/ scripts/ mcp_server/
          else
            echo "[INFO] master does not yet contain bench/; this is the bootstrap PR. Perf gate skipped."
          fi
      - name: Compare and gate
        shell: bash
        env:
          PR_LABELS: ${{ join(github.event.pull_request.labels.*.name, ',') }}
        run: |
          if [ -f master.json ] && [ -f branch.json ]; then
            python scripts/check_perf_regression.py master.json branch.json
          else
            echo "[INFO] Master bench unavailable on first run; gate is non-blocking."
          fi

  # ════════════════════════════════════════════════════════════════════════
  # SUMMARY
  # ════════════════════════════════════════════════════════════════════════

  summary:
    name: "Quality Gate Summary"
    runs-on: ubuntu-latest
    if: always()
    needs:
      # Pillar 1 — Security
      - bandit
      - semgrep
      - pip-audit
      - gitleaks
      # Pillar 3 — Memory
      - memory-baseline
      # Pillar 4 — Versatility
      - versatility-presets
      - versatility-locale
      - versatility-formats
      - property-fuzz
      # Pillar 5 — Scalability (blocking from v3.9.0)
      - perf-bench
      # Pillar 2 — Stability (blocking from v3.9.0)
      - test-count
      # Pillar 6 — Versioning
      - version-sync
      - api-surface
      - backwards-compat
      # Pillar 7 — Quality (blocking)
      - mypy
      - interrogate
    steps:
      - name: Aggregate results
        shell: bash
        run: |
          echo "## Quality Gate Status" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "| Pillar | Job | Result |" >> $GITHUB_STEP_SUMMARY
          echo "|--------|-----|--------|" >> $GITHUB_STEP_SUMMARY
          echo "| 1 Security | Bandit SAST | ${{ needs.bandit.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 1 Security | Semgrep | ${{ needs.semgrep.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 1 Security | pip-audit | ${{ needs.pip-audit.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 1 Security | Gitleaks | ${{ needs.gitleaks.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 3 Memory | Baseline tests | ${{ needs.memory-baseline.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 4 Versatility | Preset matrix | ${{ needs.versatility-presets.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 4 Versatility | Locale matrix | ${{ needs.versatility-locale.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 4 Versatility | Format matrix | ${{ needs.versatility-formats.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 4 Versatility | Property-based fuzz | ${{ needs.property-fuzz.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 2 Stability | Test count guard | ${{ needs.test-count.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 5 Scalability | Perf gate 10% | ${{ needs.perf-bench.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 6 Versioning | Version sync | ${{ needs.version-sync.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 6 Versioning | API surface | ${{ needs.api-surface.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 6 Versioning | Backwards-compat | ${{ needs.backwards-compat.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 7 Quality | mypy strict | ${{ needs.mypy.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| 7 Quality | Docstring coverage | ${{ needs.interrogate.result }} |" >> $GITHUB_STEP_SUMMARY
          # `skipped` is acceptable (PR-only jobs running on push events skip cleanly).
          # `cancelled` is acceptable (concurrency cancel of older runs).
          # Only `failure` blocks the gate.
          for r in \
            "${{ needs.bandit.result }}" \
            "${{ needs.semgrep.result }}" \
            "${{ needs.pip-audit.result }}" \
            "${{ needs.gitleaks.result }}" \
            "${{ needs.memory-baseline.result }}" \
            "${{ needs.versatility-presets.result }}" \
            "${{ needs.versatility-locale.result }}" \
            "${{ needs.versatility-formats.result }}" \
            "${{ needs.property-fuzz.result }}" \
            "${{ needs.test-count.result }}" \
            "${{ needs.perf-bench.result }}" \
            "${{ needs.version-sync.result }}" \
            "${{ needs.api-surface.result }}" \
            "${{ needs.backwards-compat.result }}" \
            "${{ needs.mypy.result }}" \
            "${{ needs.interrogate.result }}"; do
            if [[ "$r" == "failure" ]]; then
              echo "" >> $GITHUB_STEP_SUMMARY
              echo "❌ One or more pillars failed. Address the issues above before review." >> $GITHUB_STEP_SUMMARY
              exit 1
            fi
          done
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "✅ All blocking Quality Gate pillars passed." >> $GITHUB_STEP_SUMMARY
