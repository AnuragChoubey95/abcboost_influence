#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to parse 'statistics.txt' and create a single composite grouped bar chart
showing the *averaged* loss increase across all datasets for each percentage, 
comparing BoostIn vs LCA.

Dependencies:
  - matplotlib
  - numpy
  - re (for regex)
"""

import re
import numpy as np
import matplotlib.pyplot as plt


def parse_statistics_file(filepath):
    """
    Parses 'statistics.txt' and returns a nested dict:
    data[dataset_name][percentage_str] = {
       'BoostIn': float_value,
       'LCA':     float_value
    }
    """
    data = {}

    # Regex patterns to detect lines
    dataset_header_re = re.compile(r"^Substring:\s+([^|]+)\|\|")
    percentage_re = re.compile(r"^\s+Percentage\s+([\d\.]+%)\s*:\s*$")
    boostin_val_re = re.compile(r"^\s+BoostIn Average Loss Increase:\s+([-?\d\.]+)")
    lca_val_re = re.compile(r"^\s+LCA Average Loss Increase:\s+([-?\d\.]+)")

    current_dataset = None
    current_percentage = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            # Detect the start of a dataset block
            m = dataset_header_re.search(line)
            if m:
                current_dataset = m.group(1).strip()
                if current_dataset not in data:
                    data[current_dataset] = {}
                continue

            # Detect lines starting with "Percentage x.x%:"
            m = percentage_re.search(line)
            if m:
                current_percentage = m.group(1).strip()
                if current_percentage not in data[current_dataset]:
                    data[current_dataset][current_percentage] = {'BoostIn': None, 'LCA': None}
                continue

            # Detect lines with "BoostIn Average Loss Increase: xxx"
            m = boostin_val_re.search(line)
            if m and current_dataset and current_percentage:
                val = float(m.group(1))
                data[current_dataset][current_percentage]['BoostIn'] = val
                continue

            # Detect lines with "LCA Average Loss Increase: xxx"
            m = lca_val_re.search(line)
            if m and current_dataset and current_percentage:
                val = float(m.group(1))
                data[current_dataset][current_percentage]['LCA'] = val
                continue

    return data


def compute_average_loss_increase(data_dict, all_percentages=None):
    """
    Computes the average BoostIn and LCA loss increases across *all datasets*
    for each percentage in 'all_percentages'.

    Returns two dicts:
      avg_boostin[percentage] = average_value
      avg_lca[percentage]     = average_value
    Only includes percentages actually found in data.
    """
    if all_percentages is None:
        all_percentages = ["0.1%", "0.5%", "1.0%", "1.5%", "2.0%"]

    avg_boostin = {}
    avg_lca = {}

    # Initialize accumulators
    count_perc = {p: 0 for p in all_percentages}
    sum_boostin = {p: 0.0 for p in all_percentages}
    sum_lca = {p: 0.0 for p in all_percentages}

    # Gather sums for each percentage across all datasets
    for dataset_name, stats in data_dict.items():
        for p in all_percentages:
            if p in stats:
                b_val = stats[p]["BoostIn"]
                l_val = stats[p]["LCA"]
                # If either is None, skip
                if b_val is None or l_val is None:
                    continue
                sum_boostin[p] += b_val
                sum_lca[p] += l_val
                count_perc[p] += 1

    # Compute averages
    for p in all_percentages:
        if count_perc[p] > 0:
            avg_boostin[p] = sum_boostin[p] / count_perc[p]
            avg_lca[p]     = sum_lca[p] / count_perc[p]
        else:
            # In case no dataset had that percentage
            avg_boostin[p] = None
            avg_lca[p]     = None

    return avg_boostin, avg_lca


def plot_composite_bar_chart(avg_boostin, avg_lca, out_file="composite_relabel_bars.png"):
    """
    Creates a single grouped bar chart for all percentages, with two bars
    (BoostIn, LCA) per percentage. Saves the figure as out_file.
    """

    sz = 9
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 16+sz,
        'axes.titlesize': 20,
        'axes.labelsize': 22,
        'xtick.labelsize': 13+sz,
        'ytick.labelsize': 13+sz,
        'legend.fontsize': 16,
    })
    # We'll rely on the standard percentages in ascending order
    all_percentages = ["0.1%", "0.5%", "1.0%", "1.5%", "2.0%"]

    # Filter out any percentages that had None or missing
    filtered_percs = []
    boostin_vals = []
    lca_vals = []
    for p in all_percentages:
        if avg_boostin[p] is not None and avg_lca[p] is not None:
            filtered_percs.append(p)
            boostin_vals.append(avg_boostin[p])
            lca_vals.append(avg_lca[p])

    x_positions = np.arange(len(filtered_percs))
    bar_width   = 0.4

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x_positions - bar_width/2, boostin_vals, 
           bar_width, label="BoostIn")
    ax.bar(x_positions + bar_width/2, lca_vals, 
           bar_width, label="LCA")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(filtered_percs)
    ax.set_ylabel("Average Loss Increase")
    ax.set_title("Composite Single-Test Relabel Results")
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.legend()

    fig.subplots_adjust(left=0.17, right=0.97, top=0.88, bottom=0.15)

    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    # 1) Parse the data from statistics.txt
    stats_file = "statistics.txt"
    data_dict = parse_statistics_file(stats_file)

    # 2) Compute average loss increase for each percentage
    all_perc = ["0.1%", "0.5%", "1.0%", "1.5%", "2.0%"]
    avg_boostin, avg_lca = compute_average_loss_increase(data_dict, all_perc)

    # 3) Plot a single grouped bar chart for these averaged values
    plot_composite_bar_chart(avg_boostin, avg_lca, out_file="charts/composite_relabel_bars.png")


if __name__ == "__main__":
    main()
