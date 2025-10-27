import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('../../data/phase_1.csv')

# Prepare features and labels (keeping the same sampling method as OurMethod.py)
numeric_features = data[['Proj_Id', 'Proj_Open', 'CurPriority', 'Sum_Len', 'Desc_Len',
                         'Rel_Num', 'Rel_PCNum', 'Rel_PCPercent', 'Rel_PAve', 'Rel_PMed',
                         'Rep_Num', 'Rep_PCNum', 'Rep_PCPercent', 'Rep_PAve', 'Rep_PMed']].values

labels = data['Changed']

# Apply the same K-means sampling method as in OurMethod.py
kmeans = KMeans(n_clusters=(data['Changed'] == 1).sum())
kmeans.fit(numeric_features[labels == 0])

cluster_centers = kmeans.cluster_centers_
distances = [np.linalg.norm(numeric_features[labels == 0][kmeans.labels_ == i] - center, axis=1) for i, center in
             enumerate(cluster_centers)]
nearest_indices = [np.argsort(dist)[:1] for dist in distances]
selected_samples = np.concatenate(
    [numeric_features[labels == 0][kmeans.labels_ == i][idx] for i, idx in enumerate(nearest_indices)])

minority_samples = numeric_features[labels == 1]
combined_samples = np.vstack((selected_samples, minority_samples))
combined_labels = np.concatenate((np.zeros(len(selected_samples)), np.ones(len(minority_samples))))

# Load RoBERTa model for text features
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base')

def get_roberta_features(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Get text features for selected samples
selected_texts = data.loc[data['Changed'] == 0].iloc[np.concatenate([idx for idx in nearest_indices])]['Sum_Content']
minority_texts = data.loc[data['Changed'] == 1]['Sum_Content']
combined_texts = pd.concat([selected_texts, minority_texts])

text_features = get_roberta_features(combined_texts.tolist())

# Combine numeric and text features
combined_features = np.hstack((combined_samples, text_features))

# Split data
X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_labels, test_size=0.2, random_state=42)

# Define classifiers to test
classifiers = [
    {
        'name': 'Logistic Regression',
        'classifier': LogisticRegression,
        'params': {'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 100, 'random_state': 42}
    },
    {
        'name': 'Decision Tree',
        'classifier': DecisionTreeClassifier,
        'params': {'criterion': 'gini', 'splitter': 'best', 'max_depth': 8, 'random_state': 42}
    },
    {
        'name': 'Random Forest',
        'classifier': RandomForestClassifier,
        'params': {'n_estimators': 100, 'criterion': 'gini', 'max_features': 'sqrt', 'random_state': 42}
    },
    {
        'name': 'Gaussian Naive Bayes',
        'classifier': GaussianNB,
        'params': {'priors': None, 'var_smoothing': 1e-9}
    },
    {
        'name': 'KNN',
        'classifier': KNeighborsClassifier,
        'params': {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'minkowski'}
    },
    {
        'name': 'SVM',
        'classifier': SVC,
        'params': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True, 'random_state': 42}
    },
    {
        'name': 'XGBoost',
        'classifier': XGBClassifier,
        'params': {'max_depth': 6, 'learning_rate': 0.3, 'n_estimators': 100, 
                  'objective': 'binary:logistic', 'use_label_encoder': False, 
                  'eval_metric': 'mlogloss', 'random_state': 42}
    }
]

# Results storage
results = []

print("Testing different classifiers with K-means sampling...")
print("=" * 60)

for classifier_config in classifiers:
    print(f"\nTesting: {classifier_config['name']}")
    
    try:
        # Create classifier instance
        clf = classifier_config['classifier'](**classifier_config['params'])
        
        # Train and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate AUC (handle cases where predict_proba might not be available)
        try:
            if hasattr(clf, 'predict_proba'):
                auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            else:
                auc = 0.0
        except:
            auc = 0.0
        
        # Store results
        results.append({
            'Model': classifier_config['name'],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        })
        
        print(f"F1-Score: {f1:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"AUC: {auc:.3f}")
        
    except Exception as e:
        print(f"Error with {classifier_config['name']}: {str(e)}")
        results.append({
            'Model': classifier_config['name'],
            'Accuracy': 0,
            'Precision': 0,
            'Recall': 0,
            'F1-Score': 0,
            'AUC': 0,
            'Error': str(e)
        })

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(results_df[['Model', 'F1-Score', 'Accuracy', 'Precision', 'Recall', 'AUC']].to_string(index=False))

# Sort by F1-Score to see best performing model
print("\n" + "=" * 60)
print("RESULTS SORTED BY F1-SCORE")
print("=" * 60)
sorted_results = results_df.sort_values('F1-Score', ascending=False)
print(sorted_results[['Model', 'F1-Score', 'Accuracy', 'Precision', 'Recall', 'AUC']].to_string(index=False))

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print(f"\nResults saved to: model_comparison_results.csv")

# Print the best model
best_model = sorted_results.iloc[0]
print(f"\nBest performing model: {best_model['Model']} with F1-Score: {best_model['F1-Score']:.3f}")
