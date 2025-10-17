import pandas as pd
from scipy.stats import mannwhitneyu
from itertools import combinations
import numpy as np

# --- 1. 数据加载 ---
# 假设CSV文件在当前目录下
try:
    df = pd.read_csv("apache_projects_with_priority_prob.csv")
except FileNotFoundError:
    print("错误：'apache_projects_with_priority_prob.csv' 文件未找到。请确保文件与脚本在同一目录下。")
    exit()

# --- 2. 定义待检验的指标和分组 ---
metrics_to_test = ['Precision', 'Recall', 'F1-score', 'F1-weighted', 'F1-macro']
groups = df['Group'].unique()
group_pairs = list(combinations(groups, 2))

# --- 3. 执行检验并存储详细结果 ---
# 创建一个列表来存储每一行的结果字典
all_results = []

for metric in metrics_to_test:
    for group1, group2 in group_pairs:
        # 提取两个组的数据
        data1 = df[df['Group'] == group1][metric].dropna()
        data2 = df[df['Group'] == group2][metric].dropna()

        n1 = len(data1)
        n2 = len(data2)

        # 初始化结果字典
        result_row = {
            'Metric': metric,
            'Comparison': f"{group1} vs {group2}",
            'U-statistic': np.nan,
            'p-value': np.nan,
            'Effect Size (r)': np.nan
        }

        # Mann-Whitney U检验要求每组至少有一个样本
        if n1 > 0 and n2 > 0:
            try:
                # 执行检验
                u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')

                # 计算效应量 (Rank-Biserial Correlation)
                # r = 1 - (2U / n1*n2)
                effect_size_r = 1 - (2 * u_stat) / (n1 * n2)

                result_row['U-statistic'] = u_stat
                result_row['p-value'] = p_value
                result_row['Effect Size (r)'] = effect_size_r

            except ValueError:
                # 如果一个组的所有值都相同，可能会抛出ValueError
                result_row['p-value'] = 1.0

        all_results.append(result_row)

# --- 4. 格式化并打印结果 ---
# 将结果列表转换为DataFrame
results_df = pd.DataFrame(all_results)


# 格式化p值和添加显著性标记
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


# 应用格式化
results_df['U-statistic'] = results_df['U-statistic'].map('{:.1f}'.format)
results_df['Effect Size (r)'] = results_df['Effect Size (r)'].map('{:.2f}'.format)
results_df['p-value'] = results_df['p-value'].apply(format_p_value)

# 设置索引以按指标分组显示
results_df.set_index(['Metric', 'Comparison'], inplace=True)

print("--- Mann-Whitney U Test: Detailed Results for Pairwise Group Comparison ---")
print("Effect Size (r) interpretation: |r|<0.1 (trivial), 0.1-0.3 (small), 0.3-0.5 (medium), >0.5 (large)")
print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
print("-" * 75)

# 打印格式化后的DataFrame
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
    print(results_df)