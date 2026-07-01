name: Release

on:
  release:
    types: [published]

permissions:
  contents: read

jobs:
  test:
    name: Pre-release Tests
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install package + test deps
        # Same install list as ci.yml — collect-only loads every test
        # module including those that depend on hypothesis / psutil / numpy
        # / pyyaml. Without these the run errors out during collection.
        shell: bash
        env:
          HF_HUB_OFFLINE: "1"
          TRANSFORMERS_OFFLINE: "1"
          HF_HUB_DISABLE_PROGRESS_BARS: "1"
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-rerunfailures hypothesis psutil numpy pyyaml
          pytest tests/ -v

  build:
    name: Build Distribution
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
      - uses: actions/upload-artifact@v7
        with:
          name: python-dist
          path: dist/
          retention-days: 5

  publish-pypi:
    name: Publish to PyPI
    needs: build
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v8
        with:
          name: python-dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1

  publish-npm:
    name: Publish to NPM
    needs: publish-pypi
    runs-on: ubuntu-latest
    environment: release
    permissions:
      contents: read
      id-token: write
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-node@v6
        with:
          node-version: "20"
          registry-url: "https://registry.npmjs.org"
      - name: Sync version from release tag
        run: |
          VERSION="${GITHUB_REF_NAME#v}"
          cd npm
          node -e "
            const pkg = require('./package.json');
            pkg.version = '${VERSION}';
            require('fs').writeFileSync('package.json', JSON.stringify(pkg, null, 2) + '\n');
          "
      - name: Copy LICENSE and README
        run: |
          cp LICENSE npm/LICENSE
          cp README.md npm/README.md
      - name: Publish to NPM
        working-directory: npm
        run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
          NPM_CONFIG_PROVENANCE: "false"

  publish-docker:
    name: Push Docker Image
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v7
      - uses: docker/setup-buildx-action@v4
      - uses: docker/login-action@v4
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/metadata-action@v6
        id: meta
        with:
          images: ghcr.io/lyonzin/knowledge-rag
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=raw,value=latest
      - uses: docker/build-push-action@v7
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
