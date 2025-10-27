import pandas as pd
from scipy.stats import mannwhitneyu
from itertools import combinations

df = pd.read_csv("apache_projects_with_priority_prob.csv")

df['Language'] = df['Language'].str.strip()  # Ensure no leading/trailing spaces


# --- 2. Custom grouping logic ---
def assign_language_group(language):
    if language == 'Java':
        return 'Java Only'
    elif language.startswith('Java'):
        return 'Java-centric (Multi-language)'
    else:
        return 'Other Languages'


df['LanguageGroup'] = df['Language'].apply(assign_language_group)

# Print grouping results to ensure correct grouping
print("--- Project Count per Language Group ---")
print(df['LanguageGroup'].value_counts())
print("\n")

# --- 3. Define metrics and groups to be tested ---
metrics_to_test = ['Precision', 'Recall', 'F1-score', 'F1-weighted', 'F1-macro']
groups = df['LanguageGroup'].unique()
group_pairs = list(combinations(groups, 2))

# --- 4. Execute tests and store results ---
results = {}
for metric in metrics_to_test:
    p_values = {}
    for group1, group2 in group_pairs:
        # Extract data from two groups
        data1 = df[df['LanguageGroup'] == group1][metric].dropna()
        data2 = df[df['LanguageGroup'] == group2][metric].dropna()

        # Perform Mann-Whitney U test
        if len(data1) > 1 and len(data2) > 1:  # Need at least 2 samples to compare distributions
            try:
                stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                p_values[f"{group1} vs {group2}"] = p_value
            except ValueError:
                p_values[f"{group1} vs {group2}"] = 1.0
        else:
            p_values[f"{group1} vs {group2}"] = float('nan')  # Insufficient samples

    results[metric] = p_values

# --- 5. Format and print results ---
results_df = pd.DataFrame(results).sort_index()


def highlight_significant(val):
    if pd.notna(val) and val < 0.05:
        return 'background-color: yellow; font-weight: bold;'
    return ''


styled_df = results_df.style.applymap(highlight_significant).format("{:.4f}")

print("--- Mann-Whitney U Test Results (p-values) for Pairwise Language Group Comparison ---")
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
    print(styled_df.to_string())
