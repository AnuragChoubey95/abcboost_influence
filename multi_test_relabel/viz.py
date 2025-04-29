
"""
Parse statistics.txt (multi‑test relabel format) and create
• one grouped‑bar chart per dataset
• one composite chart averaged across all datasets
with consistent large fonts and generous padding.

Dependencies
------------
matplotlib, numpy
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt


SZ       = 5.0                # master font‑size offset
FIGSIZE  = (9, 5)            # width, height in inches
PAD      = dict(left=0.17, right=0.97, top=0.88, bottom=0.15)

plt.rcParams.update({
    'font.size':      16 + SZ,
    'axes.titlesize': 20,
    'axes.labelsize': 15+SZ,
    'xtick.labelsize': 13 + SZ,
    'ytick.labelsize': 13 + SZ,
    'legend.fontsize': 16,
})


def parse_statistics_file(statistics_path):
    """
    Return:
        data[dataset][relabel_int] = {'BoostIn': float, 'LCA': float}
    """
    data = {}
    cur_ds, cur_pct = None, None

    with open(statistics_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith("Dataset:"):
                cur_ds = line.split(":", 1)[1].strip()
                data[cur_ds] = {}
                continue

            if line.startswith("---") and "Relabel" in line:
                # e.g. "--- 15% Relabel ---"
                try:
                    cur_pct = int(re.findall(r"(\d+)", line)[0])
                    data[cur_ds][cur_pct] = {"BoostIn": 0.0, "LCA": 0.0}
                except (ValueError, IndexError):
                    cur_pct = None
                continue

            if line.startswith("BoostIn:") and cur_ds and cur_pct is not None:
                data[cur_ds][cur_pct]["BoostIn"] = float(line.split(":", 1)[1])
                continue

            if line.startswith("LCA:") and cur_ds and cur_pct is not None:
                data[cur_ds][cur_pct]["LCA"] = float(line.split(":", 1)[1])

    return data


def save_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def make_axes():
    """Return a fresh (fig, ax) pair with unified figsize."""
    return plt.subplots(figsize=FIGSIZE)


def plot_individual_datasets(data, outdir="charts"):
    save_dir(outdir)

    for ds, res in data.items():
        relabels = sorted(res)
        if not relabels:
            continue

        boost = [res[p]["BoostIn"] for p in relabels]
        lca   = [res[p]["LCA"]     for p in relabels]
        labels = [f"{p}%" for p in relabels]
        xpos   = np.arange(len(relabels))
        width  = 0.45

        fig, ax = make_axes()
        ax.bar(xpos - width/2, boost, width, label="BoostIn")
        ax.bar(xpos + width/2, lca,   width, label="LCA")

        ax.set_xlabel("Relabel Percentage")
        ax.set_ylabel("Average Delta Value")
        ax.set_title(f"Multi‑Test Relabel: {ds}")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.subplots_adjust(**PAD)
        fig.savefig(os.path.join(outdir, f"{ds}_multi_relabel.png"), dpi=150)
        plt.close(fig)


def plot_composite_chart(data, out_path="charts/composite_multi_relabel.png"):
    save_dir(os.path.dirname(out_path))

    # Gather all % levels present
    all_pcts = sorted({pct for ds in data.values() for pct in ds})

    boost_avg, lca_avg = [], []
    for pct in all_pcts:
        b_sum = l_sum = cnt = 0
        for ds in data.values():
            if pct in ds:
                b_sum += ds[pct]["BoostIn"]
                l_sum += ds[pct]["LCA"]
                cnt   += 1
        boost_avg.append(b_sum / cnt)
        lca_avg.append(l_sum / cnt)

    labels = [f"{p}%" for p in all_pcts]
    xpos   = np.arange(len(all_pcts))
    width  = 0.45

    fig, ax = make_axes()
    ax.bar(xpos - width/2, boost_avg, width, label="BoostIn")
    ax.bar(xpos + width/2, lca_avg,   width, label="LCA")

    ax.set_xlabel("Relabel Percentage")
    ax.set_ylabel("Average Delta Value")
    ax.set_title("Multi‑Test Relabel: Composite")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.subplots_adjust(**PAD)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    STATS_PATH = "statistics.txt"       # adjust if needed

    data = parse_statistics_file(STATS_PATH)
    plot_individual_datasets(data)
    plot_composite_chart(data)

    print("All charts have been generated with unified fonts and padding.")
