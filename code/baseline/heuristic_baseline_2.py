import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


def create_and_evaluate_heuristic_baseline_phase2(filepath, random_seed=42):
    """
    Creates and evaluates a simple probabilistic heuristic baseline for Phase II.
    The prediction is based on the historical transition probability distribution
    for each initial priority.

    Args:
        filepath (str): Path to the phase_2_modified.xlsx file.
        random_seed (int): Seed for the random number generator for reproducibility.
    """
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    # --- 1. 计算每个初始优先级的转移概率分布 ---
    print("--- 1. Learning Transition Probability Distribution from Data ---")

    # 使用 crosstab 计算频次，然后按行（初始优先级）进行归一化，得到概率
    transition_matrix = pd.crosstab(df['CurPriority'], df['TargetPriority'], normalize='index')

    print("Transition Probability Matrix (P(Target | Current)):")
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(transition_matrix)
    print("\n" + "=" * 80 + "\n")

    # --- 2. 构建并应用概率性预测逻辑 ---
    print(f"--- 2. Applying Probabilistic Prediction (Random Seed={random_seed}) ---")
    np.random.seed(random_seed)

    # 获取所有可能的目标优先级类别
    target_classes = transition_matrix.columns.to_numpy()

    def probabilistic_predict_phase2(row):
        # 获取当前行的初始优先级
        current_priority = row['CurPriority']

        try:
            # 从转移矩阵中获取该初始优先级的概率分布
            probabilities = transition_matrix.loc[current_priority].to_numpy()

            # 使用 np.random.choice 根据指定的概率分布进行随机抽样
            # size=1 表示只抽取一个结果
            return np.random.choice(target_classes, size=1, p=probabilities)[0]

        except KeyError:
            # 如果测试集中出现了训练集中没有的初始优先级，则随机猜一个
            return np.random.choice(target_classes, size=1)[0]

    # 将预测函数应用到DataFrame的每一行
    df['HeuristicPrediction'] = df.apply(probabilistic_predict_phase2, axis=1)

    print("Distribution of heuristic predictions:")
    print(df['HeuristicPrediction'].value_counts(normalize=True).sort_index())
    print("\n" + "=" * 80 + "\n")

    # --- 3. 评估预测性能 ---
    print("--- 3. Performance Evaluation of the Phase II Heuristic Baseline ---")

    y_true = df['TargetPriority']
    y_pred = df['HeuristicPrediction']

    # 打印关键的多分类指标
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    f1_weighted = report_dict['weighted avg']['f1-score']
    f1_macro = report_dict['macro avg']['f1-score']

    print(f"Overall Accuracy:  {accuracy:.4f}")
    print(f"F1-score (Macro):  {f1_macro:.4f} (Treats each class equally)")
    print(f"F1-score (Weighted): {f1_weighted:.4f} (Considers class imbalance)\n")

    # 打印完整的分类报告以供详细分析
    print("--- Full Classification Report ---")
    # 确保所有标签都存在于报告中
    labels = sorted(list(set(y_true) | set(y_pred)))
    print(classification_report(y_true, y_pred, zero_division=0, digits=4, labels=labels))


# --- 使用示例 ---
if __name__ == "__main__":
    # 请将文件名替换为您的实际文件名
    file_path_phase2 = "phase_2_modified.xlsx"
    create_and_evaluate_heuristic_baseline_phase2(file_path_phase2, random_seed=42)