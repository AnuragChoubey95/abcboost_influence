
"""
Create per‑dataset and composite bar charts showing mislabeled‑count detection
(BoostIn vs LCA) with unified large fonts and generous padding.

Dependencies
------------
matplotlib, numpy, re
"""

import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt


SZ       = 5                # master font‑size offset
FIGSIZE  = (9, 5)            # width × height in inches
PAD      = dict(left=0.17, right=0.97, top=0.88, bottom=0.15)

plt.rcParams.update({
    'font.size':      16 + SZ,
    'axes.titlesize': 20,
    'axes.labelsize': 15+SZ,
    'xtick.labelsize': 13 + SZ,
    'ytick.labelsize': 13 + SZ,
    'legend.fontsize': 16,
})


def parse_mislabel_stats(filepath):
    """
    Returns
        data[dataset]['BoostIn'][percent] = count
        data[dataset]['LCA'][percent]     = count
    """
    data = {}
    cur_ds = cur_meth = None

    with open(filepath, encoding="utf-8") as file:
        for line in map(str.strip, file):
            if line.startswith("Dataset:"):
                cur_ds = line.split(":", 1)[1].strip()
                data[cur_ds] = {"BoostIn": {}, "LCA": {}}
            elif line.startswith("Method:"):
                cur_meth = line.split(":", 1)[1].strip()
            elif line.startswith("Top") and "Mislabeled count:" in line:
                m = re.match(r"Top (\d+)% -> Mislabeled count: (\d+)", line)
                if m and cur_ds and cur_meth:
                    pct   = int(m.group(1))
                    count = int(m.group(2))
                    data[cur_ds][cur_meth][pct] = count
    return data


def make_axes():
    """Create a fresh (fig, ax) pair using global FIGSIZE."""
    return plt.subplots(figsize=FIGSIZE)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def plot_individual_charts(data, output_dir="charts"):
    ensure_dir(output_dir)

    for ds, methods in data.items():
        percents     = sorted(methods["BoostIn"])
        boost_vals   = [methods["BoostIn"].get(p, 0) for p in percents]
        lca_vals     = [methods["LCA"].get(p, 0)     for p in percents]

        xpos   = range(len(percents))
        width  = 0.35

        fig, ax = make_axes()
        ax.bar([i - width/2 for i in xpos], boost_vals, width, label="BoostIn")
        ax.bar([i + width/2 for i in xpos], lca_vals,   width, label="LCA")

        ax.set_xticks(list(xpos))
        ax.set_xticklabels([f"{p}%" for p in percents])
        ax.set_xlabel("Top‑Ranked Percent")
        ax.set_ylabel("Mislabeled Count")
        ax.set_title(f"Mislabeled Detection – {ds}")
        ax.legend()

        fig.subplots_adjust(**PAD)
        fig.savefig(os.path.join(output_dir, f"{ds}_fixlabel_bar.png"), dpi=150)
        plt.close(fig)


def plot_composite_chart(data, output_path="charts/composite_fixlabel_bar.png"):
    ensure_dir(os.path.dirname(output_path))

    pct_totals = defaultdict(lambda: {"BoostIn": 0, "LCA": 0, "n": 0})

    for methods in data.values():
        for pct in methods["BoostIn"]:
            pct_totals[pct]["BoostIn"] += methods["BoostIn"][pct]
            pct_totals[pct]["LCA"]     += methods["LCA"][pct]
            pct_totals[pct]["n"]       += 1

    percents   = sorted(pct_totals)
    boost_avg  = [pct_totals[p]["BoostIn"] / pct_totals[p]["n"] for p in percents]
    lca_avg    = [pct_totals[p]["LCA"]     / pct_totals[p]["n"] for p in percents]

    xpos  = range(len(percents))
    width = 0.35

    fig, ax = make_axes()
    ax.bar([i - width/2 for i in xpos], boost_avg, width, label="BoostIn")
    ax.bar([i + width/2 for i in xpos], lca_avg,   width, label="LCA")

    ax.set_xticks(list(xpos))
    ax.set_xticklabels([f"{p}%" for p in percents])
    ax.set_xlabel("Top‑Ranked Percent")
    ax.set_ylabel("Average Mislabeled Count")
    ax.set_title("Composite Mislabeled Detection Performance")
    ax.legend()

    fig.subplots_adjust(**PAD)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    STATS_PATH = "statistics.txt"

    stats = parse_mislabel_stats(STATS_PATH)
    plot_individual_charts(stats)
    plot_composite_chart(stats)
    print("Charts saved in 'charts/' with unified fonts and padding.")
