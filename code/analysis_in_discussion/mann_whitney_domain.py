import pandas as pd
from scipy.stats import mannwhitneyu
from itertools import combinations
import numpy as np

# --- 1. Data loading ---
# Assuming CSV file is in current directory
try:
    df = pd.read_csv("apache_projects_with_priority_prob.csv")
except FileNotFoundError:
    print("Error: 'apache_projects_with_priority_prob.csv' file not found. Please ensure the file is in the same directory as the script.")
    exit()

# --- 2. Define metrics and groups to be tested ---
metrics_to_test = ['Precision', 'Recall', 'F1-score', 'F1-weighted', 'F1-macro']
groups = df['Group'].unique()
group_pairs = list(combinations(groups, 2))

# --- 3. Execute tests and store detailed results ---
# Create a list to store result dictionaries for each row
all_results = []

for metric in metrics_to_test:
    for group1, group2 in group_pairs:
        # Extract data from two groups
        data1 = df[df['Group'] == group1][metric].dropna()
        data2 = df[df['Group'] == group2][metric].dropna()

        n1 = len(data1)
        n2 = len(data2)

        # Initialize result dictionary
        result_row = {
            'Metric': metric,
            'Comparison': f"{group1} vs {group2}",
            'U-statistic': np.nan,
            'p-value': np.nan,
            'Effect Size (r)': np.nan
        }

        # Mann-Whitney U test requires at least one sample per group
        if n1 > 0 and n2 > 0:
            try:
                # Execute test
                u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')

                # Calculate effect size (Rank-Biserial Correlation)
                # r = 1 - (2U / n1*n2)
                effect_size_r = 1 - (2 * u_stat) / (n1 * n2)

                result_row['U-statistic'] = u_stat
                result_row['p-value'] = p_value
                result_row['Effect Size (r)'] = effect_size_r

            except ValueError:
                # If all values in a group are the same, ValueError might be thrown
                result_row['p-value'] = 1.0

        all_results.append(result_row)

# --- 4. Format and print results ---
# Convert result list to DataFrame
results_df = pd.DataFrame(all_results)


# Format p-values and add significance markers
def format_p_value(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        stars = '***'
        p_str = '<0.001'
    elif p < 0.01:
        stars = '**'
        p_str = f"{p:.3f}"
    elif p < 0.05:
        stars = '*'
        p_str = f"{p:.3f}"
    else:
        stars = ''
        p_str = f"{p:.3f}"
    return f"{p_str}{stars}"


# Apply formatting
results_df['U-statistic'] = results_df['U-statistic'].map('{:.1f}'.format)
results_df['Effect Size (r)'] = results_df['Effect Size (r)'].map('{:.2f}'.format)
results_df['p-value'] = results_df['p-value'].apply(format_p_value)

# Set index to display grouped by metric
results_df.set_index(['Metric', 'Comparison'], inplace=True)

print("--- Mann-Whitney U Test: Detailed Results for Pairwise Group Comparison ---")
print("Effect Size (r) interpretation: |r|<0.1 (trivial), 0.1-0.3 (small), 0.3-0.5 (medium), >0.5 (large)")
print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
print("-" * 75)

# Print formatted DataFrame
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
    print(results_df)