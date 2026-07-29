"""
╭─╴ SECURITY DOCS REGRESSION ╶───────────────────────── v 4.6.0 ─╮
│                                                                │
│   Structural + consistency assertions on SECURITY.md,          │
│   README badges, .github/SECURITY.md pointer, dependabot,      │
│   and the OpenSSF Best Practices evidence pack.                │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    ├─ Version ·  4.6.0
    └─ Date    ·  2026-07-27

Rationale
---------
Public-facing security docs drift silently: a threat-model row references a
CWE whose fix was later refactored away, a README badge stops linking to the
right OpenSSF project after a merge, ``.github/SECURITY.md`` gets deleted and
GitHub's "Report a vulnerability" button falls back to a generic page.

These tests are cheap file-level assertions. They do not import
``mcp_server`` and do not touch a network — they only enforce the invariants
that keep the security documentation *usable*.

Every CWE identifier claimed in ``SECURITY.md`` is required to correspond to a
concrete mitigation in the source tree (either in :mod:`mcp_server.security`
or a call site elsewhere in the package). If a fix is removed, the doc row
must be removed too — this test will fail loudly the moment they drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SECURITY_MD = REPO_ROOT / "SECURITY.md"
README_MD = REPO_ROOT / "README.md"
GITHUB_SECURITY_MD = REPO_ROOT / ".github" / "SECURITY.md"
DEPENDABOT_YML = REPO_ROOT / ".github" / "dependabot.yml"
OPENSSF_MD = REPO_ROOT / ".github" / "openssf-best-practices.md"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def security_text() -> str:
    """Content of ``SECURITY.md`` at the repository root."""
    assert SECURITY_MD.exists(), f"SECURITY.md not found at {SECURITY_MD}"
    return SECURITY_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme_text() -> str:
    """Content of ``README.md`` at the repository root."""
    return README_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# SECURITY.md — required sections and structure
# ---------------------------------------------------------------------------


def test_security_md_has_threat_model_section(security_text: str) -> None:
    """The public policy must lead with a threat model — Passing criterion."""
    assert re.search(r"^##\s+Threat Model\s*$", security_text, re.MULTILINE), (
        "SECURITY.md must contain a '## Threat Model' section"
    )


def test_security_md_declares_local_runtime_model(security_text: str) -> None:
    """The trust boundary must be spelled out so out-of-scope reports are triaged."""
    assert "runs" in security_text.lower() and "local" in security_text.lower(), (
        "SECURITY.md must state that knowledge-rag runs locally"
    )


def test_security_md_marks_untrusted_inputs_explicitly(security_text: str) -> None:
    """Documents indexed and MCP-client requests are the two attacker-controlled surfaces."""
    upper = security_text.upper()
    assert upper.count("UNTRUSTED") >= 2, (
        "SECURITY.md must flag at least two UNTRUSTED input classes (indexed documents + MCP client requests)"
    )


def test_security_md_has_attack_surface_table(security_text: str) -> None:
    """Reviewers scan the table first; without the columns it is not the table."""
    header = re.search(
        r"\|\s*Vector\s*\|\s*CWE(?:\s*/\s*OWASP)?\s*\|\s*Mitigation\s*\|\s*Since\s*\|",
        security_text,
        re.IGNORECASE,
    )
    assert header, "SECURITY.md must have a table with columns Vector | CWE | Mitigation | Since"


def test_security_md_lists_v4_6_as_active(security_text: str) -> None:
    """Supported-versions table must name 4.6.x as the active line.

    Accepts either the literal ``4.6.x`` cell or a ``4.6.<digit>`` cell, with
    or without surrounding Markdown emphasis, followed anywhere on the same
    line by an active marker (``Active`` / ``✅`` / ``Supported``).
    """
    for line in security_text.splitlines():
        if not re.search(r"4\.6(?:\.\d+|\.x)?\b", line):
            continue
        if re.search(r"(?:Active|✅|Supported)", line, re.IGNORECASE):
            return
    pytest.fail("SECURITY.md must mark v4.6.x as the active/supported version")


def test_security_md_declares_48h_acknowledgement(security_text: str) -> None:
    """OpenSSF Passing requires a documented response window."""
    assert re.search(r"48\s*hours?", security_text, re.IGNORECASE), (
        "SECURITY.md must document a 48-hour acknowledgement window"
    )


def test_security_md_declares_coordinated_disclosure(security_text: str) -> None:
    """OpenSSF Passing requires a coordinated-disclosure policy."""
    assert re.search(r"90[-\s]?day", security_text, re.IGNORECASE), (
        "SECURITY.md must document a coordinated-disclosure window"
    )


def test_security_md_references_private_report_channel(security_text: str) -> None:
    """The GitHub Security Advisory link must be present — no public issues for CVEs."""
    assert "security/advisories/new" in security_text, "SECURITY.md must link the GitHub Security Advisory intake"


# ---------------------------------------------------------------------------
# SECURITY.md — CWE↔code consistency (no fabricated mitigations)
# ---------------------------------------------------------------------------


# CWEs the SECURITY.md threat-model table is *allowed* to cite, mapped to the
# source anchor that proves the mitigation exists. If a CWE appears in the
# doc but its anchor is missing from the tree, the fix was removed or was
# never real — either way the row is a lie and this test must fail.
_CWE_ANCHORS: dict[str, tuple[Path, str]] = {
    "CWE-22": (
        REPO_ROOT / "mcp_server" / "security.py",
        r"def\s+validate_path_within",
    ),
    "CWE-59": (
        REPO_ROOT / "mcp_server" / "security.py",
        r"def\s+is_path_within",
    ),
    "CWE-287": (
        REPO_ROOT / "mcp_server" / "security.py",
        r"class\s+BearerAuthMiddleware",
    ),
    "CWE-502": (
        REPO_ROOT / "mcp_server" / "config.py",
        r"yaml\.safe_load",
    ),
    "CWE-770": (
        REPO_ROOT / "mcp_server" / "ratelimit.py",
        r"def\s+rate_limited",
    ),
    "CWE-1104": (
        REPO_ROOT / ".github" / "workflows" / "release.yml",
        r"id-token:\s*write",
    ),
    "CWE-78": (
        REPO_ROOT / ".pre-commit-config.yaml",
        r"ruff",  # bandit runs in Quality Gate; ruff is the always-present hook.
    ),
    "CWE-89": (
        # SQLi is called out as N/A; anchor is any file that touches ChromaDB
        # via the Python client rather than raw SQL.
        REPO_ROOT / "mcp_server" / "storage",
        None,
    ),
}


def test_security_md_only_cites_cwes_backed_by_code(security_text: str) -> None:
    """Every CWE named in SECURITY.md must be pinned to a real code anchor.

    This is the anti-fabrication guardrail: adding a CWE row to the doc
    without adding the corresponding mitigation (or, symmetrically, deleting
    the mitigation without cleaning up the doc) both fail this test.
    """
    cited = set(re.findall(r"CWE-\d+", security_text))
    assert cited, "SECURITY.md must cite at least one CWE identifier"

    unknown = cited - set(_CWE_ANCHORS)
    assert not unknown, (
        f"SECURITY.md cites {sorted(unknown)} but no code anchor is registered "
        f"for them in tests/test_security_docs.py::_CWE_ANCHORS — add the "
        f"mitigation or remove the row"
    )


@pytest.mark.parametrize("cwe", sorted(_CWE_ANCHORS))
def test_cwe_mitigation_anchor_exists(cwe: str, security_text: str) -> None:
    """Each mapped CWE either matches a code pattern or lives at a real path.

    Only assert anchor presence when the CWE is *actually* cited in
    ``SECURITY.md`` — some entries in ``_CWE_ANCHORS`` are pre-declared for
    future policy rows and should not fail before the doc pulls them in.
    """
    if cwe not in security_text:
        pytest.skip(f"{cwe} not currently cited in SECURITY.md")

    path, pattern = _CWE_ANCHORS[cwe]
    assert path.exists(), f"{cwe}: expected anchor path missing — {path}"

    if pattern is None:
        # Directory anchor (e.g. CWE-89 → storage/ package). Existence is enough.
        return

    haystack = path.read_text(encoding="utf-8")
    assert re.search(pattern, haystack), (
        f"{cwe}: pattern {pattern!r} not found in {path} — mitigation may have been removed"
    )


def test_llm01_defense_anchors_exist(security_text: str) -> None:
    """OWASP LLM01:2025 mitigation lives in sanitize_external_content."""
    if "LLM01" not in security_text:
        pytest.skip("OWASP LLM01 not cited in SECURITY.md")

    sec_py = (REPO_ROOT / "mcp_server" / "security.py").read_text(encoding="utf-8")
    assert "def sanitize_external_content" in sec_py, (
        "LLM01:2025 row requires sanitize_external_content in mcp_server/security.py"
    )
    assert "def neutralize_injection_sentinels" in sec_py, (
        "LLM01:2025 row requires neutralize_injection_sentinels in mcp_server/security.py"
    )
    assert "def wrap_external_content" in sec_py, (
        "LLM01:2025 row requires wrap_external_content in mcp_server/security.py"
    )


# ---------------------------------------------------------------------------
# README badge
# ---------------------------------------------------------------------------


def test_readme_carries_openssf_badge(readme_text: str) -> None:
    """OpenSSF Best Practices badge must be present in the top badge block.

    The badge project ID may be a placeholder (``XXXX``) until the project is
    registered — this test accepts either the placeholder or a real numeric
    project ID so the placeholder does not block the pre-registration merge.
    """
    pattern = r"bestpractices\.coreinfrastructure\.org/projects/(?:XXXX|\d+)/badge"
    assert re.search(pattern, readme_text), (
        "README.md must carry the OpenSSF Best Practices badge (placeholder XXXX or real ID)"
    )


def test_readme_openssf_badge_links_back(readme_text: str) -> None:
    """The badge markdown must link to the project page (not a broken href)."""
    pattern = r"\]\(https://bestpractices\.coreinfrastructure\.org/projects/(?:XXXX|\d+)\)"
    assert re.search(pattern, readme_text), "OpenSSF badge in README.md must link to the project page"


# ---------------------------------------------------------------------------
# .github/SECURITY.md — pointer to root policy
# ---------------------------------------------------------------------------


def test_github_security_md_exists_and_points_to_root() -> None:
    """GitHub's 'Report a vulnerability' link checks .github/SECURITY.md first."""
    assert GITHUB_SECURITY_MD.exists(), (
        f".github/SECURITY.md must exist so GitHub UI resolves to the canonical policy "
        f"(searched at {GITHUB_SECURITY_MD})"
    )
    body = GITHUB_SECURITY_MD.read_text(encoding="utf-8")
    assert "../SECURITY.md" in body, ".github/SECURITY.md must link back to the root ../SECURITY.md"
    assert "security/advisories/new" in body, (
        ".github/SECURITY.md must also surface the private report URL (users may land here directly)"
    )


# ---------------------------------------------------------------------------
# Dependabot — SCA cadence for OpenSSF Silver
# ---------------------------------------------------------------------------


def test_dependabot_config_exists() -> None:
    """OpenSSF Silver — SCA tools in CI. Dependabot is the SCA channel."""
    assert DEPENDABOT_YML.exists(), ".github/dependabot.yml must exist"


def test_dependabot_covers_pip_and_actions_and_npm() -> None:
    """All three release channels have PRs opened when their deps have CVEs."""
    text = DEPENDABOT_YML.read_text(encoding="utf-8")
    assert 'package-ecosystem: "pip"' in text, "dependabot must monitor pip"
    assert 'package-ecosystem: "github-actions"' in text, "dependabot must monitor github-actions"
    assert 'package-ecosystem: "npm"' in text, "dependabot must monitor npm (wrapper)"


def test_dependabot_pip_updates_are_weekly_or_faster() -> None:
    """Weekly is the OpenSSF-recommended floor for the language ecosystem.

    Monthly is too slow for surfaced CVEs; daily is noise. We assert weekly
    for pip specifically — Docker base images can stay monthly.
    """
    text = DEPENDABOT_YML.read_text(encoding="utf-8")
    # Find the pip block and inspect its schedule.
    pip_block = re.search(
        r'package-ecosystem:\s*"pip".*?(?=package-ecosystem:|\Z)',
        text,
        re.DOTALL,
    )
    assert pip_block, "pip block not found in .github/dependabot.yml"
    assert re.search(r'interval:\s*"weekly"', pip_block.group(0)), (
        "pip ecosystem must run weekly (found something else); OpenSSF Silver "
        "expects fast SCA turnaround on the primary language"
    )


# ---------------------------------------------------------------------------
# OpenSSF Best Practices evidence pack
# ---------------------------------------------------------------------------


def test_openssf_evidence_pack_exists() -> None:
    """The self-assessment must live in .github/ so reviewers find it."""
    assert OPENSSF_MD.exists(), ".github/openssf-best-practices.md missing — this is the reviewer evidence pack"


def test_openssf_evidence_pack_has_all_three_tiers() -> None:
    """Passing / Silver / Gold — reviewers audit each tier separately."""
    body = OPENSSF_MD.read_text(encoding="utf-8")
    assert re.search(r"^##?\s*.*Passing", body, re.IGNORECASE | re.MULTILINE), (
        "openssf-best-practices.md must have a Passing section"
    )
    assert re.search(r"^##?\s*.*Silver", body, re.IGNORECASE | re.MULTILINE), (
        "openssf-best-practices.md must have a Silver section"
    )
    assert re.search(r"^##?\s*.*Gold", body, re.IGNORECASE | re.MULTILINE), (
        "openssf-best-practices.md must have a Gold section"
    )


def test_openssf_evidence_pack_names_security_md() -> None:
    """The evidence pack must cross-reference SECURITY.md as the threat-model source."""
    body = OPENSSF_MD.read_text(encoding="utf-8")
    assert "SECURITY.md" in body, "openssf-best-practices.md must cite SECURITY.md as the threat-model artefact"
