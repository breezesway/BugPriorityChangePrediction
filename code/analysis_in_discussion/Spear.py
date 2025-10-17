import pandas as pd
from scipy.stats import spearmanr
import itertools

# 读取数据
df = pd.read_csv("apache_projects_with_priority_prob.csv")

# 定义两组变量
group_A = ["PriorityChangeProb", "Age(y)", "#Revision", "#Committer", "#Bug in JIRA", "#Bug with priority changes"]
group_B = ["Precision", "Recall", "F1-score", "F1-weighted", "F1-macro"]

results = []

# 两两 Spearman 相关性
for a, b in itertools.product(group_A, group_B):
    corr, p_value = spearmanr(df[a], df[b])
    results.append({
        "Var_A": a,
        "Var_B": b,
        "Spearman_corr": corr,
        "p_value": p_value
    })

df_results = pd.DataFrame(results)

# 格式化：p值如果大于 0.001 就用小数点后 4 位，否则保留科学计数法
df_results["p_value"] = df_results["p_value"].apply(
    lambda x: f"{x:.4f}" if x >= 0.001 else f"{x:.2e}"
)

# 保存结果
df_results.to_csv("spearman_results_formatted.csv", index=False)

print(df_results)
