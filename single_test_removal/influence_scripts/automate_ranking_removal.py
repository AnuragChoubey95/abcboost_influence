#!/usr/bin/env python3

import sys
import os
import re
import random
import numpy as np

from rank_train_samples import process_infl_scores  # function to rank based on col
from generate_custom_data import filter_data        # function to remove top p%

def load_task_from_map(dataset_substring, map_file="dataset_task_map.sh"):
    """
    Reads 'dataset_task_map.sh' line by line to find the line matching:
      [<dataset_substring>]="<task>"
    Returns the task string (binary, multiclass, regression) or None if not found.
    """
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"{map_file} not found.")

    pattern = re.compile(r'^\s*\[' + re.escape(dataset_substring) + r'\]="([^"]+)"')

    with open(map_file, "r") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                return match.group(1).strip()
    return None


def determine_method(task):
    """
    From the task (binary, multiclass, regression), return a tuple:
      (method_str, method_suffix)
    For regression, method_str='regression' and method_suffix='_p2'; otherwise suffix=''
    
    e.g. 'binary'     -> ('robustlogit', '')
         'multiclass' -> ('mart', '')
         'regression' -> ('regression', '_p2')
    """
    if task == "binary":
        return ("robustlogit", "")
    elif task == "multiclass":
        return ("mart", "")
    elif task == "regression":
        return ("regression", "_p2")
    else:
        raise ValueError(f"Unknown task type: {task}")


def build_influence_filepath(substring, method_str, method_suffix, is_lca=False):
    """
    Build final CSV path for either BoostIn or LCA, e.g.:
    compas.test.csv_robustlogit_J20_v0.1BoostIn_Influence.csv
    or for regression:
    compas.test.csv_regression_J20_v0.1_p2LCA_Influence.csv

    If is_lca=True, we add 'LCA' else 'BoostIn'.
    """
    influence_type = "LCA" if is_lca else "BoostIn"
    return f"../../influence_scores/{substring}.test.csv_{method_str}_J20_v0.1{method_suffix}{influence_type}_Influence.csv"


def get_num_columns_from_csv(csv_path):
    """
    Reads the first line of the CSV and returns how many columns there are.
    Raises an error if 0 columns or file not found.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Influence file not found: {csv_path}")

    with open(csv_path, "r") as f:
        first_line = f.readline().rstrip("\n")
    columns = len(first_line.split(","))
    if columns == 0:
        raise ValueError("Unable to determine number of columns from CSV.")
    return columns


def rank_influence_for_index(influence_csv, col_idx, output_prefix, subdir):
    """
    1) Decide if CSV is LCA or BoostIn by checking 'LCA' in the path
    2) Build output name e.g. {prefix}_column_{col_idx}_BoostIn.csv
    3) Call process_infl_scores(influence_csv, col_idx, out_file)
    4) Return the ranking-file path
    """
    if "LCA" in influence_csv:
        suffix = "_LCA"
    else:
        suffix = "_BoostIn"

    os.makedirs(subdir, exist_ok=True)
    out_file = os.path.join(subdir, f"{output_prefix}_column_{col_idx}{suffix}.csv")
    process_infl_scores(influence_csv, col_idx, out_file)
    return out_file


def remove_top_percent(ranking_file, data_file, pct_str, out_dir):
    """
    Calls filter_data(...) from generate_custom_data using the same arguments.
    Example:
      remove_top_percent('compas_ranked_column_0_BoostIn.csv', 'compas.train.csv','1.5%','compas')
    """
    filter_data(
        ranking_file = ranking_file,
        data_file    = data_file,
        percentage   = pct_str,
        output_dir   = out_dir
    )

def extract_test_sample_row(test_file, row_idx, out_dir, prefix):
    """
    Extracts a single row (row_idx) from the test_file and saves it to:
    {out_dir}/{prefix}_test_column_{row_idx}.csv
    """
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    with open(test_file, "r") as f:
        lines = f.readlines()

    if row_idx >= len(lines):
        raise IndexError(f"Row index {row_idx} out of bounds for test file with {len(lines)} lines.")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_test_column_{row_idx}.csv")

    with open(out_path, "w") as f_out:
        f_out.write(lines[row_idx])

    return out_path


def rank_and_remove_influential_samples(
    boostin_csv,
    lca_csv,
    data_file,
    substring,
    ranked_dir,
    indices,
    percentages
):
    """
    Given:
      - boostin_csv: Path to the BoostIn CSV
      - lca_csv: Path to the LCA CSV (may not exist)
      - data_file: Path to the original .train.csv
      - substring: dataset identifier
      - ranked_dir: directory to save the ranking files
      - indices: list of column indices to rank over
      - percentages: e.g. [0.1, 0.5, 1, 1.5, 2]

    For each col_idx in indices:
      1) rank_influence_for_index(...) with boostin_csv
      2) rank_influence_for_index(...) with lca_csv if present
      3) for each percentage in percentages:
           remove_top_percent(...) for both BoostIn & LCA
    """
    has_lca = (lca_csv and os.path.isfile(lca_csv))

    test_file = f"../../data/{substring}.test.csv"
    test_out_dir = f"../test_data/{substring}"

    for col_idx in indices:
        print(f"\nProcessing column index: {col_idx}")

        # Extract the test row corresponding to this column index
        
        try:
            test_sample_path = extract_test_sample_row(
                test_file=test_file,
                row_idx=col_idx,
                out_dir=test_out_dir,
                prefix=substring
            )
            print(f"Saved test sample to {test_sample_path}")
        except Exception as e:
            print(f"Warning: Could not extract test row {col_idx}: {e}")

        out_file_boostin = rank_influence_for_index(
            influence_csv = boostin_csv,
            col_idx       = col_idx,
            output_prefix = f"{substring}_ranked",
            subdir        = ranked_dir
        )

        out_file_lca = rank_influence_for_index(
            influence_csv = lca_csv,
            col_idx       = col_idx,
            output_prefix = f"{substring}_ranked",
            subdir        = ranked_dir
        )

        for p in percentages:
            # e.g. '1.5' => '1.5%'
            if isinstance(p, float) and not str(p).endswith('%'):
                p_str = f"{p}%"
            else:
                p_str = str(p)
                if not p_str.endswith('%'):
                    p_str += "%"

            print(f"Removing top {p_str} from {data_file} using {out_file_boostin} / {out_file_lca} if present")

            remove_top_percent(
                ranking_file=out_file_boostin,
                data_file=data_file,
                pct_str=p_str,
                out_dir= f"../custom_data/{substring}"
            )

            remove_top_percent(
                ranking_file=out_file_lca,
                data_file=data_file,
                pct_str=p_str,
                out_dir= f"../custom_data/{substring}"
            )

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 automate_ranking_removal.py <dataset_substring>")
        sys.exit(1)

    substring = sys.argv[1]
    map_file = "dataset_task_map.sh"

    try:
        task = load_task_from_map(substring, map_file)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    if not task:
        print(f"Error: Could not find valid entry for '{substring}' in {map_file}.")
        sys.exit(1)

    print(f"Found dataset '{substring}' with task type '{task}'.")

    try:
        method_str, method_suffix = determine_method(task)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print(f"Will build influence filenames for method={method_str}, suffix={method_suffix}")

    boostin_csv = build_influence_filepath(substring, method_str, method_suffix, is_lca=False)
    lca_csv     = build_influence_filepath(substring, method_str, method_suffix, is_lca=True)

    try:
        num_cols = get_num_columns_from_csv(boostin_csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error with BoostIn CSV: {e}")
        sys.exit(1)

    print(f"Number of columns in {boostin_csv}: {num_cols}")

    has_lca = os.path.isfile(lca_csv)
    if not has_lca:
        print(f"Warning: LCA CSV not found: {lca_csv}. We'll proceed with just BoostIn.")

    random.seed(42)
    indices = random.sample(range(num_cols), 100)

    percentages = [0.1, 0.5, 1, 1.5, 2]

    data_file = f"../../data/{substring}.train.csv"
    ranked_dir = "../ranked_rows"

    rank_and_remove_influential_samples(
        boostin_csv=boostin_csv,
        lca_csv=lca_csv if has_lca else None,
        data_file=data_file,
        substring=substring,
        ranked_dir=ranked_dir,
        indices=indices,
        percentages=percentages
    )

    print("\nRankings Complete for indices:", indices)
    print("Removals Complete for percentages:", percentages)
    print("Custom data sets have been prepared!")

if __name__ == "__main__":
    main()
