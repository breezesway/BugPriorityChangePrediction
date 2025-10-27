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

    # --- 1. Calculate transition probability distribution for each initial priority ---
    print("--- 1. Learning Transition Probability Distribution from Data ---")

    # Use crosstab to calculate frequencies, then normalize by row (initial priority) to get probabilities
    transition_matrix = pd.crosstab(df['CurPriority'], df['TargetPriority'], normalize='index')

    print("Transition Probability Matrix (P(Target | Current)):")
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(transition_matrix)
    print("\n" + "=" * 80 + "\n")

    # --- 2. Build and apply probabilistic prediction logic ---
    print(f"--- 2. Applying Probabilistic Prediction (Random Seed={random_seed}) ---")
    np.random.seed(random_seed)

    # Get all possible target priority classes
    target_classes = transition_matrix.columns.to_numpy()

    def probabilistic_predict_phase2(row):
        # Get the initial priority of the current row
        current_priority = row['CurPriority']

        try:
            # Get the probability distribution for this initial priority from the transition matrix
            probabilities = transition_matrix.loc[current_priority].to_numpy()

            # Use np.random.choice to randomly sample according to the specified probability distribution
            # size=1 means only sample one result
            return np.random.choice(target_classes, size=1, p=probabilities)[0]

        except KeyError:
            # If the test set contains initial priorities not present in the training set, randomly guess one
            return np.random.choice(target_classes, size=1)[0]

    # Apply the prediction function to each row of the DataFrame
    df['HeuristicPrediction'] = df.apply(probabilistic_predict_phase2, axis=1)

    print("Distribution of heuristic predictions:")
    print(df['HeuristicPrediction'].value_counts(normalize=True).sort_index())
    print("\n" + "=" * 80 + "\n")

    # --- 3. Evaluate prediction performance ---
    print("--- 3. Performance Evaluation of the Phase II Heuristic Baseline ---")

    y_true = df['TargetPriority']
    y_pred = df['HeuristicPrediction']

    # Print key multi-classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    f1_weighted = report_dict['weighted avg']['f1-score']
    f1_macro = report_dict['macro avg']['f1-score']

    print(f"Overall Accuracy:  {accuracy:.4f}")
    print(f"F1-score (Macro):  {f1_macro:.4f} (Treats each class equally)")
    print(f"F1-score (Weighted): {f1_weighted:.4f} (Considers class imbalance)\n")

    # Print complete classification report for detailed analysis
    print("--- Full Classification Report ---")
    # Ensure all labels exist in the report
    labels = sorted(list(set(y_true) | set(y_pred)))
    print(classification_report(y_true, y_pred, zero_division=0, digits=4, labels=labels))


# --- Usage example ---
if __name__ == "__main__":
    # Please replace the filename with your actual filename
    file_path_phase2 = "phase_2.csv"
    create_and_evaluate_heuristic_baseline_phase2(file_path_phase2, random_seed=42)