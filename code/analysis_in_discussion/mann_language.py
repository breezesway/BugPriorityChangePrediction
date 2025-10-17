import pandas as pd
from scipy.stats import mannwhitneyu
from itertools import combinations

df = pd.read_csv("apache_projects_with_priority_prob.csv")

df['Language'] = df['Language'].str.strip()  # 确保没有前导/尾随空格


# --- 2. 自定义分组逻辑 ---
def assign_language_group(language):
    if language == 'Java':
        return 'Java Only'
    elif language.startswith('Java'):
        return 'Java-centric (Multi-language)'
    else:
        return 'Other Languages'


df['LanguageGroup'] = df['Language'].apply(assign_language_group)

# 打印分组结果，确保分组正确
print("--- Project Count per Language Group ---")
print(df['LanguageGroup'].value_counts())
print("\n")

# --- 3. 定义待检验的指标和分组 ---
metrics_to_test = ['Precision', 'Recall', 'F1-score', 'F1-weighted', 'F1-macro']
groups = df['LanguageGroup'].unique()
group_pairs = list(combinations(groups, 2))

# --- 4. 执行检验并存储结果 ---
results = {}
for metric in metrics_to_test:
    p_values = {}
    for group1, group2 in group_pairs:
        # 提取两个组的数据
        data1 = df[df['LanguageGroup'] == group1][metric].dropna()
        data2 = df[df['LanguageGroup'] == group2][metric].dropna()

        # 执行Mann-Whitney U检验
        if len(data1) > 1 and len(data2) > 1:  # 至少需要2个样本才能比较分布
            try:
                stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                p_values[f"{group1} vs {group2}"] = p_value
            except ValueError:
                p_values[f"{group1} vs {group2}"] = 1.0
        else:
            p_values[f"{group1} vs {group2}"] = float('nan')  # 样本不足

    results[metric] = p_values

# --- 5. 格式化并打印结果 ---
results_df = pd.DataFrame(results).sort_index()


def highlight_significant(val):
    if pd.notna(val) and val < 0.05:
        return 'background-color: yellow; font-weight: bold;'
    return ''


styled_df = results_df.style.applymap(highlight_significant).format("{:.4f}")

print("--- Mann-Whitney U Test Results (p-values) for Pairwise Language Group Comparison ---")
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
    print(styled_df.to_string())
