import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


def create_and_evaluate_heuristic_cross_dataset(filepath, custom_probabilities=None, random_seed=42):
    """
    Evaluates a probabilistic heuristic baseline.
    It can either learn probabilities from data or use a predefined custom set.

    Args:
        filepath (str): Path to the phase_1.xlsx file.
        custom_probabilities (dict, optional): A predefined dictionary of change probabilities.
                                               If None, probabilities are learned from data.
        random_seed (int): Seed for reproducibility.
    """
    try:
        df_original = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    # --- 1. 定义条件概率 ---
    if custom_probabilities:
        print("--- 1. Using PREDEFINED Custom Conditional Probabilities ---")
        change_probabilities = custom_probabilities
    else:
        # 如果未提供自定义概率，则从数据中学习
        print("--- 1. Learning Conditional Probabilities from ORIGINAL Dataset ---")
        change_probabilities = df_original.groupby('CurPriority')['Changed'].mean().to_dict()

    print("Change Probability Mapping (CurPriority -> Probability):")
    for priority, prob in change_probabilities.items():
        print(f"  Priority {priority}: {prob:.4f}")
    print("\n" + "=" * 50 + "\n")

    # --- 2. 创建一个平衡的测试集 (代码不变) ---
    print("--- 2. Creating a Balanced TEST Dataset via Undersampling ---")
    df_majority = df_original[df_original.Changed == 0]
    df_minority = df_original[df_original.Changed == 1]
    n_minority = len(df_minority)
    df_majority_downsampled = df_majority.sample(n=n_minority, random_state=random_seed)
    df_balanced_test = pd.concat([df_majority_downsampled, df_minority])
    df_balanced_test = df_balanced_test.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print(f"Balanced test set size: {len(df_balanced_test)}")
    print("\n" + "=" * 50 + "\n")

    # --- 3. 将定义的规则应用到平衡测试集上 (代码不变) ---
    print(f"--- 3. Applying Defined Rule to Balanced Test Set (Random Seed={random_seed}) ---")
    np.random.seed(random_seed)

    def probabilistic_predict(row):
        prob_change = change_probabilities.get(row['CurPriority'], 0)
        return 1 if np.random.rand() < prob_change else 0

    df_balanced_test['HeuristicPrediction'] = df_balanced_test.apply(probabilistic_predict, axis=1)
    print("Distribution of heuristic predictions on the balanced test set:")
    print(df_balanced_test['HeuristicPrediction'].value_counts(normalize=True))
    print("\n" + "=" * 50 + "\n")

    # --- 4. 评估性能 (代码不变) ---
    print("--- 4. Performance Evaluation of Heuristic Baseline (on Balanced Test Set) ---")
    y_true = df_balanced_test['Changed']
    y_pred = df_balanced_test['HeuristicPrediction']
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    accuracy = report_dict['accuracy']
    weighted_precision = report_dict['weighted avg']['precision']
    weighted_recall = report_dict['weighted avg']['recall']
    weighted_f1 = report_dict['weighted avg']['f1-score']
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1-score:  {weighted_f1:.4f}")


# --- 使用示例 ---
if __name__ == "__main__":
    file_path_phase1 = "phase_1.xlsx"

    # --- 核心修改在这里 ---
    # 定义您想要的自定义概率
    # 注意：键(key)应该是整数或浮点数，与您 'CurPriority' 列中的数据类型一致
    # 假设 'CurPriority' 列是整数 1, 2, 3, 4, 5
    my_custom_probs = {
        1: 0.9638,
        2: 0.8283,
        3: 0.493,
        4: 0.2687,
        5: 0.44  # 审稿意见中第五个类别未指定，我假设您是指类别5
    }

    # 将自定义概率字典作为参数传入函数
    create_and_evaluate_heuristic_cross_dataset(
        file_path_phase1,
        custom_probabilities=my_custom_probs,
        random_seed=42
    )