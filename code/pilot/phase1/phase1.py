import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the data
data = pd.read_excel(r'E:\GitClone\BugPriorityChangePrediction\data\phase_1.xlsx')

# Separate features and labels
numeric_features = data[['Proj_Id', 'Proj_Open', 'CurPriority', 'Sum_Len', 'Desc_Len',
                         'Rel_Num', 'Rel_PCNum', 'Rel_PCPercent', 'Rel_PAve', 'Rel_PMed',
                         'Rep_Num', 'Rep_PCNum', 'Rep_PCPercent', 'Rep_PAve', 'Rep_PMed']].values
labels = data['Changed']

# ==============================================================================
# STEP 1: Perform Random Oversampling (as done previously)
# ==============================================================================

# Find indices for majority and minority classes
majority_indices = np.where(labels == 0)[0]
minority_indices = np.where(labels == 1)[0]

# Set seed for reproducibility
np.random.seed(42)

# Perform random OVERsampling of the minority class
resampled_minority_indices = np.random.choice(minority_indices, size=len(majority_indices), replace=True)

# Combine the original majority samples with the newly oversampled minority samples
# This creates a large, balanced dataset
selected_samples = np.vstack((numeric_features[majority_indices], numeric_features[resampled_minority_indices]))
combined_labels = np.concatenate((np.zeros(len(majority_indices)), np.ones(len(resampled_minority_indices))))


# ==============================================================================
# NEW STEP: Take a 10% random subset of the balanced dataset
# ==============================================================================

# Calculate the size of the 10% subset
total_balanced_samples = len(selected_samples)
subset_size = int(total_balanced_samples * 0.1)

# Generate random indices for the subset
subset_indices = np.random.choice(np.arange(total_balanced_samples), size=subset_size, replace=False)

# Create the final, smaller subset of samples and labels
final_samples = selected_samples[subset_indices]
final_labels = combined_labels[subset_indices]

# NOTE: We print the shape to confirm the size reduction
print(f"Original balanced dataset size: {selected_samples.shape[0]}")
print(f"Final subset size for training (10%): {final_samples.shape[0]}")


# ==============================================================================
# Final step: Use the 10% subset for training and testing
# ==============================================================================

# Split the 10% SUBSET into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(final_samples, final_labels, test_size=0.2, random_state=42)

# Initialize classifiers
rf_clf = RandomForestClassifier(max_depth=10, n_estimators=280, random_state=42, min_samples_split=5, min_samples_leaf=2)
knn_clf = KNeighborsClassifier(n_neighbors=385, metric='manhattan', weights='distance')
svc_clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=9, learning_rate=0.01, n_estimators=95, subsample=1.0, colsample_bytree=0.5, min_child_weight=1)

# Initialize the soft voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_clf),
        ('knn', knn_clf),
        ('svc', svc_clf),
        ('xgb', xgb_clf)
    ],
    voting='soft',
    weights=[4, 1, 1, 4]
)

# Train the model
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

# Evaluate the model
print("\n--- Model Evaluation Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1]))