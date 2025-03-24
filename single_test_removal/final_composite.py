import re
import numpy as np

def parse_statistics_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    boostin_values = []
    lca_values = []
    
    for line in data:
        match = re.search(r"BoostIn Average Loss Increase: ([\d\.-]+)", line)
        if match:
            boostin_values.append(float(match.group(1)))
        
        match = re.search(r"LCA Average Loss Increase: ([\d\.-]+)", line)
        if match:
            lca_values.append(float(match.group(1)))
    
    # Compute composite scores by averaging
    boostin_composite_score = np.mean(boostin_values) if boostin_values else 0
    lca_composite_score = np.mean(lca_values) if lca_values else 0
    
    return boostin_composite_score, lca_composite_score

# Example usage
file_path = "statistics.txt"  # Replace with the actual file path
boostin_score, lca_score = parse_statistics_file(file_path)

print(f"BoostIn Composite Score: {boostin_score:.6f}")
print(f"LCA Composite Score: {lca_score:.6f}")
