import os
import re
import pandas as pd

def load_dataset_task_map(file_path="dataset_task_map.sh"):
    """
    Parse the dataset_task_map.sh file returning a dict
    that maps dataset_substring -> task_type.
    Example lines:
      [compas]="binary"
      [dry_bean]="multiclass"
      [concrete]="regression"
    """
    dataset_task_map = {}
    pattern = re.compile(r'\s*\[(.+?)\]=\"(.+?)\"')
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                dataset, task = match.groups()
                dataset_task_map[dataset] = task
    return dataset_task_map

def check_mislabelling(ranked_orig_file, ranked_mis_file):
    """
    Given two CSVs with the same shape and row alignment:
      - ranked_orig_file: original training data with 'rank' col
      - ranked_mis_file: mislabelled training data with 'rank' col
    Both have the label in column 0. We check how many rows differ
    in column 0. Then for each p in [5,10,15,20,25,30], we see
    how many such mislabeled examples appear in the top p% by rank.
    Returns a dict p->count_of_mislabeled_in_top_p.
    """
    orig_df = pd.read_csv(ranked_orig_file)
    mis_df = pd.read_csv(ranked_mis_file)

    n = len(orig_df)
    if len(mis_df) != n:
        raise ValueError(f"Mismatch in #rows: original={n}, mislabelled={len(mis_df)}")

    # Ensure both have 'rank' col
    if 'rank' not in orig_df.columns or 'rank' not in mis_df.columns:
        raise ValueError("Missing 'rank' column in one of the dataframes.")

    # We'll store the label difference as a boolean for easy counting
    # If col 0 is different => that row was mislabelled
    label_diff = (orig_df.iloc[:, 0] != mis_df.iloc[:, 0])

    results = {}
    for p in range(5, 31, 5):  # 5,10,15,20,25,30
        cutoff = int(n * (p / 100.0))
        # top p% => rank <= cutoff
        # Because the rank goes 1..n, rank=1 means top row
        # We'll find all rows in mis_df with rank <= cutoff
        top_rows = mis_df[mis_df["rank"] <= cutoff]

        # Among these top rows, how many are mislabeled?
        # Because row alignment hasn't changed, we can do:
        top_mislabel_count = label_diff[top_rows.index].sum()

        results[p] = int(top_mislabel_count)
    return results

def main():
    dataset_task_map = load_dataset_task_map("dataset_task_map.sh")
    ranked_dir = os.path.join("..", "ranked_training")
    methods = ["BoostIn", "LCA"]
    stats_file = "../statistics.txt"

    # We'll open stats_file in append mode
    with open(stats_file, "a") as f_out:
        for dataset_substring, task_type in dataset_task_map.items():
            f_out.write(f"\n========================================\n")
            f_out.write(f"Dataset: {dataset_substring}\n")
            for method_type in methods:
                # Build the file names
                orig_ranked = os.path.join(ranked_dir, f"{dataset_substring}_{method_type}_original_ranked.csv")
                mis_ranked = os.path.join(ranked_dir, f"{dataset_substring}_{method_type}_mislabel_ranked.csv")

                if not os.path.exists(orig_ranked) or not os.path.exists(mis_ranked):
                    f_out.write(f"[WARN] Missing one of the ranked files for {dataset_substring}, {method_type}\n")
                    continue

                # Count mislabelled in top p%
                try:
                    stats_dict = check_mislabelling(orig_ranked, mis_ranked)
                except Exception as e:
                    f_out.write(f"[ERROR] {e}\n")
                    continue

                f_out.write(f"  Method: {method_type}\n")
                for p, count_mis in stats_dict.items():
                    f_out.write(f"    Top {p}% -> Mislabeled count: {count_mis}\n")

if __name__ == "__main__":
    main()
