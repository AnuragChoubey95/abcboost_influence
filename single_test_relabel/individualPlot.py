#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to parse 'statistics.txt' and create grouped bar charts showing average
loss increases at different relabel percentages for each dataset, comparing
BoostIn vs LCA.

Dependencies:
  - matplotlib
  - numpy

Usage:
  python visualize_relabel_stats.py
"""

import re
import numpy as np
import matplotlib.pyplot as plt


def parse_statistics_file(filepath):
    """
    Parses the statistics.txt file and returns a nested dict:
    data[dataset_name][percentage_str] = {
       'BoostIn': float_value,
       'LCA':     float_value
    }
    """
    data = {}

    # Regex patterns to detect lines
    substring_header_re = re.compile(r"^Substring:\s+([^|]+)\|\|")
    percentage_re = re.compile(r"^\s+Percentage\s+([\d\.]+%)\s*:\s*$")
    boostin_val_re = re.compile(r"^\s+BoostIn Average Loss Increase:\s+([-?\d\.]+)")
    lca_val_re = re.compile(r"^\s+LCA Average Loss Increase:\s+([-?\d\.]+)")

    current_dataset = None
    current_percentage = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            # Check for start of a dataset block
            m = substring_header_re.search(line)
            if m:
                current_dataset = m.group(1).strip()
                # Initialize if not present
                if current_dataset not in data:
                    data[current_dataset] = {}
                continue
            
            # Check for line that starts with "Percentage x.x%:"
            m = percentage_re.search(line)
            if m:
                current_percentage = m.group(1).strip()
                # Initialize structure for this percentage
                if current_percentage not in data[current_dataset]:
                    data[current_dataset][current_percentage] = {'BoostIn': None, 'LCA': None}
                continue
            
            # Check for BoostIn line
            m = boostin_val_re.search(line)
            if m and current_dataset and current_percentage:
                val = float(m.group(1))
                data[current_dataset][current_percentage]['BoostIn'] = val
                continue
            
            # Check for LCA line
            m = lca_val_re.search(line)
            if m and current_dataset and current_percentage:
                val = float(m.group(1))
                data[current_dataset][current_percentage]['LCA'] = val
                continue

    return data


def plot_relabel_bars(data_dict, outdir='charts'):
    """
    Creates grouped bar charts for each dataset in data_dict.
    data_dict format: data[dataset_name][percentage_str] = {'BoostIn': val, 'LCA': val}
    Saves one PNG per dataset in the 'outdir' directory.
    """
    import os
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # For consistent ordering of percentages
    all_percentages = ["0.1%", "0.5%", "1.0%", "1.5%", "2.0%"]

    for dataset_name, stats in data_dict.items():
        # We'll gather the data in the order of all_percentages if present
        x_labels = [p for p in all_percentages if p in stats]
        if not x_labels:
            # If no recognized percentages, skip
            continue

        # Collect y-values for each method in a consistent order
        boostin_vals = [stats[p]["BoostIn"] for p in x_labels]
        lca_vals     = [stats[p]["LCA"]     for p in x_labels]

        x_positions = np.arange(len(x_labels))
        bar_width = 0.4

        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot bars
        ax.bar(x_positions - bar_width/2, boostin_vals, 
               bar_width, label="BoostIn")
        ax.bar(x_positions + bar_width/2, lca_vals, 
               bar_width, label="LCA")

        # X-axis labels, ticks
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Average Loss Increase")
        ax.set_title(f"Single Test Relabel - {dataset_name.strip()}")

        # Optional zero-line
        ax.axhline(0, color='gray', linewidth=0.8)

        ax.legend()
        fig.tight_layout()

        outpath = os.path.join(outdir, f"{dataset_name.strip()}_relabel_bar.png")
        plt.savefig(outpath, dpi=150)
        plt.close(fig)


def main():
    stats_file = "statistics.txt"  # adjust path if needed
    data = parse_statistics_file(stats_file)
    plot_relabel_bars(data, outdir='charts')


if __name__ == "__main__":
    main()
