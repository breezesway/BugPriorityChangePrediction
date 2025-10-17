import pandas as pd
from scipy.stats import mannwhitneyu


def analyze_priority_data_vs_baseline(filepath, baseline_group=5):
    """
    Reads a CSV, calculates feature means, and performs Mann-Whitney U tests
    comparing a specified baseline priority group against all other groups.

    Args:
        filepath (str): The path to the CSV file.
        baseline_group (int or float): The value in 'CurPriority' to be used as the baseline.
    """
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    # --- 1. 均值计算部分 (保持不变) ---
    grouping_col = 'CurPriority'
    mean_cols = [
        'Sum_Len', 'Desc_Len', 'Rel_Num', 'Rel_PCNum', 'Cmt_Num',
        'Cmt_PersNum', 'Cmt_Len', 'His_Num', 'His_PersNum', 'His_Field'
    ]

    required_cols_mean = [grouping_col] + mean_cols
    if not all(col in df.columns for col in required_cols_mean):
        missing = [col for col in required_cols_mean if col not in df.columns]
        print(f"Error for Mean Calculation: Missing columns -> {missing}")
        return

    for col in mean_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    grouped_means = df.groupby(grouping_col)[mean_cols].mean()

    print("--- 1. Mean of Features Grouped by Current Priority ---")
    with pd.option_context(
            'display.float_format', '{:.2f}'.format,
            'display.width', 1000,  # 增加宽度以容纳所有列
            'display.max_columns', None  # 核心改动：不限制显示的列数
    ):
        print(grouped_means)
    print("\n" + "=" * 80 + "\n")

    # --- 2. 修改后的 Mann-Whitney U 检验部分 ---
    test_cols = [
        'Sum_Len', 'Desc_Len', 'Cmt_Len'
    ]

    required_cols_test = [grouping_col] + test_cols
    if not all(col in df.columns for col in required_cols_test):
        missing = [col for col in required_cols_test if col not in df.columns]
        print(f"Error for Mann-Whitney Test: Missing columns -> {missing}")
        return

    print(f"--- 2. Mann-Whitney U Test Results (p-values) vs Baseline (Group {baseline_group}) ---")

    # 获取基准组数据
    baseline_data = df[df[grouping_col] == baseline_group]
    if baseline_data.empty:
        print(f"Error: Baseline group '{baseline_group}' not found in the data.")
        return

    # 获取其他所有组
    other_groups = sorted([g for g in df[grouping_col].dropna().unique() if g != baseline_group])

    p_value_results = {}

    for col in test_cols:
        p_values_for_col = {}
        # 循环遍历其他组
        for group in other_groups:
            comparison_data = df[df[grouping_col] == group][col].dropna()
            baseline_feature_data = baseline_data[col].dropna()

            if not comparison_data.empty and not baseline_feature_data.empty:
                try:
                    _, p_value = mannwhitneyu(comparison_data, baseline_feature_data, alternative='two-sided')
                    p_values_for_col[f"Group {int(group)} vs {int(baseline_group)}"] = p_value
                except ValueError:
                    p_values_for_col[f"Group {int(group)} vs {int(baseline_group)}"] = 1.0
            else:
                p_values_for_col[f"Group {int(group)} vs {int(baseline_group)}"] = float('nan')

        p_value_results[col] = p_values_for_col

    p_value_df = pd.DataFrame(p_value_results)

    # 高亮函数
    def highlight_significant(val):
        if pd.notna(val) and val < 0.05:
            return 'background-color: yellow; font-weight: bold;'
        return ''

    styled_df = p_value_df.style.applymap(highlight_significant).format("{:.4f}")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(styled_df.to_string())


# --- 使用示例 ---
if __name__ == "__main__":
    csv_file_path = "phase_2_modified.xlsx"
    # 调用函数，指定基准组为5
    analyze_priority_data_vs_baseline(csv_file_path, baseline_group=5)
