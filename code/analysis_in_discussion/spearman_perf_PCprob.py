import pandas as pd
from scipy.stats import spearmanr
import itertools

# Read data
df = pd.read_csv("apache_projects_with_priority_prob.csv")

# Define two groups of variables
group_A = ["PriorityChangeProb", "Age(y)", "#Revision", "#Committer", "#Bug in JIRA", "#Bug with priority changes"]
group_B = ["Precision", "Recall", "F1-score", "F1-weighted", "F1-macro"]

results = []

# Pairwise Spearman correlation
for a, b in itertools.product(group_A, group_B):
    corr, p_value = spearmanr(df[a], df[b])
    results.append({
        "Var_A": a,
        "Var_B": b,
        "Spearman_corr": corr,
        "p_value": p_value
    })

df_results = pd.DataFrame(results)

# Format: if p-value is greater than 0.001, use 4 decimal places, otherwise keep scientific notation
df_results["p_value"] = df_results["p_value"].apply(
    lambda x: f"{x:.4f}" if x >= 0.001 else f"{x:.2e}"
)

# Save results
df_results.to_csv("spearman_results_formatted.csv", index=False)

print(df_results)
