#!/usr/bin/env python3
"""Generate a star-history SVG for a GitHub repo.

Env:
  GH_TOKEN            (required) PAT with public_repo or fine-grained
                      Metadata:R+Contents:RW on the target repo.
  STAR_HISTORY_REPO   owner/repo (default: lyonzin/knowledge-rag)
  STAR_HISTORY_OUT    output SVG path (default: docs/star-history.svg)
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import requests

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

REPO = os.environ.get("STAR_HISTORY_REPO", "lyonzin/knowledge-rag")
TOKEN = os.environ.get("GH_TOKEN")
OUT = Path(os.environ.get("STAR_HISTORY_OUT", "docs/star-history.svg"))

if not TOKEN:
    sys.exit("ERROR: GH_TOKEN env var is required")

session = requests.Session()
session.headers.update(
    {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github.v3.star+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"star-history-action/{REPO}",
    }
)


def fetch_stargazers():
    page = 1
    while True:
        r = session.get(
            f"https://api.github.com/repos/{REPO}/stargazers",
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            return
        for s in batch:
            ts = s["starred_at"]
            yield datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if len(batch) < 100:
            return
        page += 1


def main():
    dates = sorted(fetch_stargazers())
    if not dates:
        sys.exit("No stargazers found")

    counts = list(range(1, len(dates) + 1))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.plot(dates, counts, linewidth=2.5, color="#dc3545")
    ax.fill_between(dates, counts, alpha=0.12, color="#dc3545")
    ax.set_title(f"Star History  {REPO}", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("GitHub Stars", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.text(0.99, 0.01, f"Updated {now}", ha="right", va="bottom", fontsize=8, color="#888")
    plt.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, format="svg", bbox_inches="tight", transparent=False)
    print(f"Wrote {OUT}  {len(dates)} stars, latest at {dates[-1].isoformat()}")


if __name__ == "__main__":
    main()
