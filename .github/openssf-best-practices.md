# OpenSSF Best Practices — evidence and limitations

Reviewed against the repository and public badge page on **2026-10-03**.
The registered project is [knowledge-rag #13864](https://www.bestpractices.dev/en/projects/13864), whose public page displayed **Passing** on that date. Keep the README badge linked to that project; the former `XXXX` registration instructions are obsolete.

The badge is a project self-assessment, not an independent code audit or a guarantee that a particular release has no defects. The governing requirements are the [official criteria](https://www.bestpractices.dev/en/criteria). Changes to code, CI, or release practices require reassessing the corresponding evidence.

## Passing — repository evidence

| Area | Evidence that can be inspected | Limits of that evidence |
|------|--------------------------------|-------------------------|
| License and contribution process | `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS` | File presence does not prove that every contribution received review |
| User and interface documentation | `README.md`, `docs/INSTALLATION.md`, `docs/CONFIGURATION.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `docs/TROUBLESHOOTING.md` | Examples must match the installed version and callable tool arguments |
| Versioning and release notes | `scripts/check_version_sync.py`, `scripts/check_changelog.py`, `CHANGELOG.md` | A branch version is not proof that a release was published |
| Private vulnerability reporting | `SECURITY.md`, `.github/SECURITY.md`, GitHub's private reporting form | A GitHub noreply email address is not a reporting mailbox |
| Automated tests | `tests/`, `.github/workflows/ci.yml` | Test count, skips, and expected failures do not prove that a mitigation is integrated |
| Static and dependency checks | `quality-gate.yml`, `security.yml`, `dependabot.yml`, `.github/gitleaks.toml` | Inspect the actual run and documented exceptions; scanner availability does not prove that all findings were fixed |
| Release publishing | `.github/workflows/release.yml` | OIDC publishing is distinct from signed Git tags, reproducible builds, and release-specific attestations |

The October audit found that some previous security claims were supported only by expected-failure integration tests. The corrected policy identifies unreleased fixes separately. Do not cite the old v4.6.0 table as evidence that directory containment and parser provenance were fully integrated in that release.

## Silver — further verification required

The repository configures a Linux/macOS/Windows × Python 3.11/3.12/3.13 test matrix and several quality checks. Verify the current run, rather than treating the existence of workflow YAML as a passing result.

The previous evidence document asserted account 2FA, protected branches, required approving reviews, signed tags, deterministic wheels, and complete dependency pinning without including verifiable evidence. Those claims are not established by this repository snapshot. In particular, `requirements.txt` and most `pyproject.toml` requirements are version ranges, not a locked dependency closure.

The configured coverage floor is 35%. mypy is a gradual check over selected modules and scripts. Several checks are informational or have explicit exceptions. Review the applicable criterion and actual settings before claiming Silver readiness.

## Gold — proposed work, not achieved status

Independent build reproducibility, release-specific SBOMs and attestations, coverage-guided native/parser fuzzing, additional maintainers, documented governance, hardened runners, and action SHA pinning need their own evidence. Their mention in a roadmap does not constitute implementation or an achieved badge level.

## Maintenance

Recheck this evidence after changes to the public API, trust boundaries, dependency policy, CI gates, and release workflow. Record the commit and run used for verification. Update the external self-assessment through its normal maintainer workflow when repository changes invalidate an answer; this audit does not automatically modify that external assessment.
