# Live Performance Dashboard

Static assets that render https://lyonzin.github.io/knowledge-rag/ — an executive-friendly, dark-mode-first performance dashboard fed by the pytest-benchmark output produced on every push to `master`.

## Files

| File | Purpose |
|---|---|
| `index.html` | Static HTML shell — hero, KPI grid, chart canvas, table, methodology, footer |
| `styles.css` | Design system (dark by default, auto light via `prefers-color-scheme`, manual toggle) |
| `dashboard.js` | Vanilla JS — loads `data.json`, categorizes benchmarks, renders Chart.js bar, table sort/filter |
| `build.py` | Transforms pytest-benchmark JSON export into the flat `data.json` consumed by `dashboard.js` |

## Local preview (no CI needed)

```bash
# 1. Produce a fresh bench payload
pytest bench/ --benchmark-only --benchmark-min-rounds=5 --benchmark-json=bench-current.json

# 2. Build the data.json alongside the static assets
mkdir -p /tmp/dashboard-preview
cp bench/dashboard/index.html   /tmp/dashboard-preview/
cp bench/dashboard/styles.css   /tmp/dashboard-preview/
cp bench/dashboard/dashboard.js /tmp/dashboard-preview/
python bench/dashboard/build.py \
    --input  bench-current.json \
    --output /tmp/dashboard-preview/data.json \
    --commit "$(git rev-parse --short HEAD)" \
    --version "$(python -c 'import mcp_server; print(mcp_server.__version__)')"

# 3. Serve locally (any static server)
python -m http.server 8000 --directory /tmp/dashboard-preview
# open http://localhost:8000
```

## Design decisions

- **Dark by default.** Executive dashboards live on external monitors and demo rooms — dark reads better in both. Light theme auto-switches via `prefers-color-scheme` and can be forced via the toggle in the header.
- **Zero build step.** No bundler, no framework, no npm. Chart.js loads from jsDelivr with SRI. Everything else is one HTTP request per file.
- **Log-scale latency bars.** Wall-time measurements use a logarithmic scale. Memory measurements are displayed in bytes and are excluded from the latency chart.
- **Category-driven KPIs.** Instead of showing every benchmark equally, the top of the page picks a representative benchmark per subsystem (search / FTS5 / reindex / indexing / memory / concurrent) and classifies it as healthy / watch / investigate against tuned thresholds.
- **Print-friendly.** Executives export dashboards. The stylesheet has a `@media print` block that hides interactive controls and adds simple borders so PDF exports are legible.

## Category classification rules

`dashboard.js` classifies each benchmark by matching its short name (last `::` segment) against regex patterns:

| Category | Pattern | Health thresholds (good / warn) |
|---|---|---|
| **Search** | `search / query / retrieval` | 10 ms / 50 ms |
| **FTS5** | `fts5 / lexical / fast[_-]?path` | 10 ms / 30 ms |
| **Reindex** | `reindex / rebuild / swap` | 10 s / 60 s |
| **Indexing** | `index / ingest / parse / chunk` | 100 ms / 500 ms |
| **Memory** | Explicit `measurement_unit: bytes` overrides name matching | Per-workload RSS budget in bytes |
| **Concurrent** | `concurrent / thread / parallel` | 100 ms / 500 ms |
| **Other** | — | 100 ms / 1 s |

New benchmark files under `bench/` are automatically categorized as long as the test name contains one of these keywords. The first matching pattern wins. Explicit byte measurements always use the memory category. A cache-hit latency benchmark remains a timing measurement rather than a RAM measurement. Timing thresholds are display heuristics, not service-level guarantees.

Memory rows read `rss_peak_delta_bytes`, `rss_retained_delta_bytes`, `rss_budget_bytes` and `rss_sampling_interval_ms` from the benchmark's `extra_info`. Each workload runs five times, sampling process RSS every 5 ms and applying an absolute budget to every run. Older artifacts without byte samples display `not measured` / `unavailable`; elapsed garbage-collection time is never relabeled as memory. Shorter allocation peaks can fall between samples, and RSS includes native allocations and allocator retention.

## Data schema (`data.json`)

```json
{
  "generated_at": "2026-08-13T21:45:12+00:00",
  "commit": "abc1234",
  "version": "4.8.5",
  "results": [
    {
      "name": "test_bench_search_hybrid",
      "measurement_unit": "nanoseconds",
      "measurement_value": 3140000,
      "workload_version": 1,
      "median_ns": 3140000,
      "stddev_ns": 210000,
      "ops": 318.5,
      "rounds": 20,
      "iqr_ratio": 0.15
    }
  ]
}
```

For memory rows, `measurement_unit` is `bytes`; `median_ns`, `stddev_ns`, `ops` and `iqr_ratio` are null. `measurement_value` contains sampled peak RSS growth or null when the legacy artifact has no byte sample. The RSS budget and sampling interval are also retained. Sorting groups measurements by unit instead of treating bytes as nanoseconds.

`dashboard.js` enriches each row with `category`, `category_label`, `category_color`, `status`, and `stddev_pct` at render time. `workload_version` identifies changes to the benchmark scenario; the CI gate reports different versions as incomparable and keeps enforcing its 10% threshold for unchanged, comparable scenarios.
