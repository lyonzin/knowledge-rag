/* =============================================================================
   knowledge-rag — Live Performance Dashboard
   Vanilla JS. Zero build step. Loads data.json produced by build.py.
   ============================================================================= */

(() => {
  'use strict';

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------

  // Category detection — maps benchmark filename patterns to a category slug + color.
  // Order matters: first match wins.
  const CATEGORY_RULES = [
    { pattern: /search|query|retrieval/i,           slug: 'search',     label: 'Search',     color: '#38bdf8' },
    { pattern: /fts5|lexical|fast[_-]?path/i,       slug: 'fts5',       label: 'FTS5',       color: '#22c55e' },
    { pattern: /reindex|rebuild|swap/i,             slug: 'reindex',    label: 'Reindex',    color: '#a78bfa' },
    { pattern: /index(?!ing)|ingest|parse|chunk/i,  slug: 'indexing',   label: 'Indexing',   color: '#f472b6' },
    { pattern: /memory|rss|cache/i,                 slug: 'memory',     label: 'Memory',     color: '#f59e0b' },
    { pattern: /concurrent|thread|parallel/i,       slug: 'concurrent', label: 'Concurrent', color: '#ef4444' },
  ];
  const CATEGORY_FALLBACK = { slug: 'other', label: 'Other', color: '#6c7a99' };

  // KPI categories in the summary section — one card per subsystem.
  const KPI_ORDER = ['search', 'fts5', 'reindex', 'indexing', 'memory', 'concurrent'];

  // Health thresholds per category (median ns → status). Tuned to executive audience.
  // If median under warn → good; under danger → warn; else → danger.
  const HEALTH_THRESHOLDS = {
    search:     { good_ns: 10_000_000,   warn_ns: 50_000_000 },   // 10ms good, 50ms warn
    fts5:       { good_ns: 10_000_000,   warn_ns: 30_000_000 },   //  10ms good, 30ms warn
    reindex:    { good_ns: 10_000_000_000, warn_ns: 60_000_000_000 }, // 10s / 60s
    indexing:   { good_ns: 100_000_000,  warn_ns: 500_000_000 },  // 100ms / 500ms
    memory:     { good_ns: 1_000_000_000, warn_ns: 5_000_000_000 },
    concurrent: { good_ns: 100_000_000,  warn_ns: 500_000_000 },
    other:      { good_ns: 100_000_000,  warn_ns: 1_000_000_000 },
  };

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------------
  // Theme toggle
  // ---------------------------------------------------------------------------

  const themeBtn = $('theme-toggle');
  const html = document.documentElement;
  const savedTheme = localStorage.getItem('kr-dashboard-theme') || 'auto';
  html.setAttribute('data-theme', savedTheme);
  updateThemeIcon();

  themeBtn.addEventListener('click', () => {
    const current = html.getAttribute('data-theme') || 'auto';
    const next = current === 'auto' ? 'light' : (current === 'light' ? 'dark' : 'auto');
    html.setAttribute('data-theme', next);
    localStorage.setItem('kr-dashboard-theme', next);
    updateThemeIcon();
    if (window.__krChart) window.__krChart.update();
  });
  function updateThemeIcon() {
    const t = html.getAttribute('data-theme');
    themeBtn.textContent = t === 'light' ? '☀️' : (t === 'dark' ? '🌙' : '🌓');
    themeBtn.title = `Theme: ${t} (click to toggle)`;
  }

  // ---------------------------------------------------------------------------
  // Number formatters
  // ---------------------------------------------------------------------------

  function fmtDuration(ns) {
    if (!isFinite(ns) || ns < 0) return '—';
    if (ns < 1_000)             return ns.toFixed(1) + ' ns';
    if (ns < 1_000_000)         return (ns / 1_000).toFixed(1) + ' µs';
    if (ns < 1_000_000_000)     return (ns / 1_000_000).toFixed(2) + ' ms';
    if (ns < 60_000_000_000)    return (ns / 1_000_000_000).toFixed(2) + ' s';
    return (ns / 60_000_000_000).toFixed(1) + ' min';
  }
  function fmtOps(ops) {
    if (!isFinite(ops) || ops < 0) return '—';
    if (ops >= 1_000_000) return (ops / 1_000_000).toFixed(2) + ' M/s';
    if (ops >= 1_000)     return (ops / 1_000).toFixed(2) + ' K/s';
    return ops.toFixed(2) + '/s';
  }
  function fmtPct(v) {
    return (v * 100).toFixed(1) + '%';
  }
  function fmtShortDate(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    return d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
  }

  // ---------------------------------------------------------------------------
  // Categorization
  // ---------------------------------------------------------------------------

  function classify(benchName) {
    for (const rule of CATEGORY_RULES) {
      if (rule.pattern.test(benchName)) return rule;
    }
    return CATEGORY_FALLBACK;
  }
  function healthStatus(category, medianNs) {
    const t = HEALTH_THRESHOLDS[category] || HEALTH_THRESHOLDS.other;
    if (medianNs <= t.good_ns) return { level: 'good',    label: 'healthy' };
    if (medianNs <= t.warn_ns) return { level: 'warn',    label: 'watch' };
    return                            { level: 'danger',  label: 'investigate' };
  }

  // ---------------------------------------------------------------------------
  // Main render
  // ---------------------------------------------------------------------------

  async function main() {
    let data;
    try {
      const res = await fetch('./data.json', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
    } catch (err) {
      renderError(err.message);
      return;
    }

    // Enrich rows with category + status
    const rows = (data.results || []).map(r => {
      const cat = classify(r.name);
      const status = healthStatus(cat.slug, r.median_ns);
      const stddev_pct = r.stddev_ns && r.median_ns ? (r.stddev_ns / r.median_ns) : null;
      return { ...r, category: cat.slug, category_label: cat.label, category_color: cat.color, status, stddev_pct };
    });

    renderHeroMeta(data, rows);
    renderKPIs(rows);
    renderChart(rows);
    renderCategoryBreakdown(rows);
    renderTable(rows);
    populateCategoryFilter(rows);
    wireTableControls(rows);
  }

  function renderError(msg) {
    $('hero-meta').innerHTML = `<span class="meta-item"><span class="meta-label">Error</span><span class="meta-value">${escapeHtml(msg)}</span></span>`;
    $('kpi-grid').innerHTML = `<div class="kpi-card"><div class="kpi-label">Data unavailable</div><div class="kpi-sub">Could not load <code>data.json</code>. Is the workflow published?</div></div>`;
    $('results-body').innerHTML = `<tr><td colspan="5" class="empty">No data available.</td></tr>`;
  }

  // ---------------------------------------------------------------------------
  // Hero meta
  // ---------------------------------------------------------------------------

  function renderHeroMeta(data, rows) {
    $('meta-updated').textContent = fmtShortDate(data.generated_at);
    $('meta-commit').textContent  = data.commit  || '—';
    $('meta-version').textContent = data.version || '—';
    $('meta-count').textContent   = rows.length.toString();
  }

  // ---------------------------------------------------------------------------
  // KPI grid (one card per subsystem, showing the fastest / most-representative
  // benchmark in that category)
  // ---------------------------------------------------------------------------

  function renderKPIs(rows) {
    const grid = $('kpi-grid');
    grid.innerHTML = '';

    // Group by category, pick the median-of-medians as representative
    const byCat = groupBy(rows, r => r.category);

    // Preserve KPI order; drop empty categories
    const orderedCats = KPI_ORDER.filter(c => byCat[c] && byCat[c].length);

    if (orderedCats.length === 0) {
      grid.innerHTML = `<div class="kpi-card"><div class="kpi-label">No categorized benchmarks</div><div class="kpi-sub">Add benchmark files under <code>bench/</code> matching the category patterns.</div></div>`;
      return;
    }

    for (const cat of orderedCats) {
      const list = byCat[cat].slice().sort((a, b) => a.median_ns - b.median_ns);
      const repr = list[Math.floor(list.length / 2)]; // median
      const status = healthStatus(cat, repr.median_ns);
      const catColor = classify(repr.name).color;

      const card = document.createElement('div');
      card.className = 'kpi-card';
      card.style.setProperty('--kpi-accent', catColor);
      card.innerHTML = `
        <div class="kpi-label">${escapeHtml(cat)}</div>
        <div class="kpi-value">${fmtDuration(repr.median_ns)}</div>
        <div class="kpi-sub">
          <span>${list.length} bench${list.length === 1 ? '' : 'es'}</span>
          <span class="kpi-status kpi-status--${status.level}">${status.label}</span>
        </div>
      `;
      grid.appendChild(card);
    }
  }

  // ---------------------------------------------------------------------------
  // Latency chart (horizontal bars, log scale)
  // ---------------------------------------------------------------------------

  function renderChart(rows) {
    const ctx = $('latency-chart');
    if (!ctx || !window.Chart) return;

    const sorted = rows.slice().sort((a, b) => a.median_ns - b.median_ns);
    const labels = sorted.map(r => r.name);
    const values = sorted.map(r => r.median_ns);
    const colors = sorted.map(r => r.category_color);

    const style = getComputedStyle(document.documentElement);
    const textColor   = style.getPropertyValue('--text-muted').trim();
    const gridColor   = style.getPropertyValue('--border').trim();
    const tooltipBg   = style.getPropertyValue('--bg-elev-2').trim();

    if (window.__krChart) window.__krChart.destroy();

    window.__krChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: colors,
          borderRadius: 4,
          barPercentage: 0.75,
          categoryPercentage: 0.85,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: tooltipBg,
            titleColor: style.getPropertyValue('--text').trim(),
            bodyColor: textColor,
            borderColor: gridColor,
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: (ctx) => {
                const r = sorted[ctx.dataIndex];
                return [
                  ` Median: ${fmtDuration(r.median_ns)}`,
                  ` Ops/sec: ${fmtOps(r.ops)}`,
                  ` Category: ${r.category_label}`,
                ];
              },
            },
          },
        },
        scales: {
          x: {
            type: 'logarithmic',
            title: { display: true, text: 'Median wall-time (log scale)', color: textColor },
            grid:  { color: gridColor },
            ticks: { color: textColor, callback: v => fmtDuration(v) },
          },
          y: {
            grid: { color: 'transparent' },
            ticks: { color: textColor, autoSkip: false, font: { size: 11 } },
          },
        },
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Category breakdown cards
  // ---------------------------------------------------------------------------

  function renderCategoryBreakdown(rows) {
    const grid = $('category-grid');
    grid.innerHTML = '';

    const byCat = groupBy(rows, r => r.category);
    const cats = Object.keys(byCat).sort();

    for (const cat of cats) {
      const list = byCat[cat];
      const medians = list.map(r => r.median_ns).sort((a, b) => a - b);
      const median = medians[Math.floor(medians.length / 2)];
      const min = medians[0];
      const max = medians[medians.length - 1];
      const color = list[0].category_color;

      const card = document.createElement('div');
      card.className = 'category-card';
      card.style.setProperty('--cat-color', color);
      card.setAttribute('role', 'button');
      card.setAttribute('tabindex', '0');
      card.innerHTML = `
        <div class="cat-name">${escapeHtml(list[0].category_label)}</div>
        <div class="cat-stats">
          <span><strong>${list.length}</strong> tests</span>
          <span>median <strong>${fmtDuration(median)}</strong></span>
          <span>range <strong>${fmtDuration(min)}</strong>–<strong>${fmtDuration(max)}</strong></span>
        </div>
      `;
      card.addEventListener('click', () => {
        $('category-filter').value = cat;
        $('category-filter').dispatchEvent(new Event('change'));
        document.getElementById('table-heading').scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); }
      });
      grid.appendChild(card);
    }
  }

  // ---------------------------------------------------------------------------
  // Results table (sortable + filterable)
  // ---------------------------------------------------------------------------

  let currentSort = { key: 'median_ns', dir: 'asc' };
  let currentRows = [];

  function renderTable(rows) {
    currentRows = rows;
    applyFilters();
    // Wire header sort clicks once
    document.querySelectorAll('#results-table th[data-sort]').forEach(th => {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        if (currentSort.key === key) {
          currentSort.dir = currentSort.dir === 'asc' ? 'desc' : 'asc';
        } else {
          currentSort = { key, dir: key === 'name' || key === 'category' ? 'asc' : 'asc' };
        }
        applyFilters();
      });
    });
  }

  function applyFilters() {
    const q   = ($('table-filter').value || '').trim().toLowerCase();
    const cat = $('category-filter').value || 'all';

    let filtered = currentRows.slice();
    if (q)          filtered = filtered.filter(r => r.name.toLowerCase().includes(q));
    if (cat !== 'all') filtered = filtered.filter(r => r.category === cat);

    filtered.sort((a, b) => {
      const va = a[currentSort.key], vb = b[currentSort.key];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      const cmp = typeof va === 'string' ? va.localeCompare(vb) : (va - vb);
      return currentSort.dir === 'asc' ? cmp : -cmp;
    });

    renderRows(filtered);
    updateSortIndicator();
  }

  function renderRows(rows) {
    const tbody = $('results-body');
    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty">No benchmarks match the current filter.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td><span class="cat-badge" style="border-color:${r.category_color};color:${r.category_color}">${escapeHtml(r.category_label)}</span></td>
        <td><code style="background:transparent;color:inherit;border:none;padding:0">${escapeHtml(r.name)}</code></td>
        <td class="numeric">${fmtDuration(r.median_ns)}</td>
        <td class="numeric">${fmtOps(r.ops)}</td>
        <td class="numeric">${r.stddev_pct != null ? fmtPct(r.stddev_pct) : '—'}</td>
      </tr>
    `).join('');
  }

  function updateSortIndicator() {
    document.querySelectorAll('#results-table th[data-sort]').forEach(th => {
      th.classList.remove('sorted-asc', 'sorted-desc');
      if (th.dataset.sort === currentSort.key) {
        th.classList.add(currentSort.dir === 'asc' ? 'sorted-asc' : 'sorted-desc');
      }
    });
  }

  function populateCategoryFilter(rows) {
    const sel = $('category-filter');
    const cats = Array.from(new Set(rows.map(r => r.category))).sort();
    for (const c of cats) {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = rows.find(r => r.category === c).category_label;
      sel.appendChild(opt);
    }
  }

  function wireTableControls(rows) {
    $('table-filter').addEventListener('input', applyFilters);
    $('category-filter').addEventListener('change', applyFilters);
  }

  // ---------------------------------------------------------------------------
  // Utils
  // ---------------------------------------------------------------------------

  function groupBy(arr, keyFn) {
    return arr.reduce((acc, item) => {
      const k = keyFn(item);
      (acc[k] ||= []).push(item);
      return acc;
    }, {});
  }

  function escapeHtml(s) {
    return String(s ?? '')
      .replace(/&/g,  '&amp;')
      .replace(/</g,  '&lt;')
      .replace(/>/g,  '&gt;')
      .replace(/"/g,  '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Boot
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
  } else {
    main();
  }
})();
