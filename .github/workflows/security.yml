name: Security

on:
  push:
    branches: [master]
  schedule:
    - cron: "0 6 * * 1"

permissions:
  security-events: write
  contents: read

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: github/codeql-action/init@v4
        with:
          languages: python
      - uses: github/codeql-action/analyze@v4

  dependency-audit:
    name: Dependency Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Install and audit
        run: |
          pip install pip-audit
          pip install -r requirements.txt
          pip-audit || echo "::warning::Dependency vulnerabilities found (see above)"
