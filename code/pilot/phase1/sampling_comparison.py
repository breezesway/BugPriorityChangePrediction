import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel
import torch
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('../../data/phase_1.csv')

# Prepare features and labels
numeric_features = data[['Proj_Id', 'Proj_Open', 'CurPriority', 'Sum_Len', 'Desc_Len',
                         'Rel_Num', 'Rel_PCNum', 'Rel_PCPercent', 'Rel_PAve', 'Rel_PMed',
                         'Rep_Num', 'Rep_PCNum', 'Rep_PCPercent', 'Rep_PAve', 'Rep_PMed']].values

labels = data['Changed']

# Load RoBERTa model for text features
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base')

def get_roberta_features(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Get text features for all samples
text_features = get_roberta_features(data['Sum_Content'].tolist())

# Combine numeric and text features
combined_features = np.hstack((numeric_features, text_features))

# Define sampling methods to test
sampling_methods = [
    {
        'name': 'No sampling',
        'method': None,
        'params': {}
    },
    {
        'name': 'Random undersampling',
        'method': RandomUnderSampler,
        'params': {'random_state': 42}
    },
    {
        'name': 'NearMiss',
        'method': NearMiss,
        'params': {'version': 1, 'n_neighbors': 3, 'sampling_strategy': 'auto'}
    },
    {
        'name': 'Tomek Links',
        'method': TomekLinks,
        'params': {'sampling_strategy': 'auto'}
    },
    {
        'name': 'ENN',
        'method': EditedNearestNeighbours,
        'params': {'n_neighbors': 3, 'kind_sel': 'all', 'sampling_strategy': 'auto'}
    },
    {
        'name': 'K-means',
        'method': 'custom_kmeans',
        'params': {'init': 'k-means++', 'n_init': 10, 'max_iter': 300, 'tol': 1e-4, 'algorithm': 'lloyd'}
    },
    {
        'name': 'Random oversampling',
        'method': RandomOverSampler,
        'params': {'random_state': 42}
    },
    {
        'name': 'SMOTE',
        'method': SMOTE,
        'params': {'sampling_strategy': 'auto', 'k_neighbors': 5, 'random_state': 42}
    }
]

def custom_kmeans_sampling(X, y, **params):
    """Custom K-means sampling implementation"""
    majority_indices = np.where(y == 0)[0]
    minority_indices = np.where(y == 1)[0]
    
    majority_features = X[majority_indices]
    minority_features = X[minority_indices]
    
    # Use K-means to cluster majority class
    n_clusters = len(minority_indices)
    kmeans = KMeans(n_clusters=n_clusters, **params)
    kmeans.fit(majority_features)
    
    # Select nearest samples to cluster centers
    cluster_centers = kmeans.cluster_centers_
    distances = [np.linalg.norm(majority_features[kmeans.labels_ == i] - center, axis=1) 
                for i, center in enumerate(cluster_centers)]
    nearest_indices = [np.argsort(dist)[:1] for dist in distances]
    selected_indices = np.concatenate([majority_indices[kmeans.labels_ == i][idx] 
                                     for i, idx in enumerate(nearest_indices)])
    
    # Combine selected majority samples with all minority samples
    final_indices = np.concatenate([selected_indices, minority_indices])
    
    return X[final_indices], y[final_indices]

# Results storage
results = []

print("Testing different sampling methods...")
print("=" * 60)

for sampling_config in sampling_methods:
    print(f"\nTesting: {sampling_config['name']}")
    
    try:
        # Apply sampling
        if sampling_config['method'] is None:
            # No sampling
            X_sampled = combined_features
            y_sampled = labels
        elif sampling_config['method'] == 'custom_kmeans':
            # Custom K-means sampling
            X_sampled, y_sampled = custom_kmeans_sampling(combined_features, labels, **sampling_config['params'])
        else:
            # Use imbalanced-learn methods
            sampler = sampling_config['method'](**sampling_config['params'])
            X_sampled, y_sampled = sampler.fit_resample(combined_features, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)
        
        # Define classifiers
        rf_clf = RandomForestClassifier(max_depth=10, n_estimators=280, random_state=42, min_samples_split=5,
                                        min_samples_leaf=2)
        knn_clf = KNeighborsClassifier(n_neighbors=385, metric='manhattan', weights='distance')
        svc_clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
        xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=9, learning_rate=0.01,
                                n_estimators=95, subsample=1.0, colsample_bytree=0.5, min_child_weight=1)
        
        # Voting classifier
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
        
        # Train and predict
        voting_clf.fit(X_train, y_train)
        y_pred = voting_clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1])
        
        # Store results
        results.append({
            'Sampling Method': sampling_config['name'],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc,
            'Train Size': len(X_train),
            'Test Size': len(X_test)
        })
        
        print(f"F1-Score: {f1:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"AUC: {auc:.3f}")
        
    except Exception as e:
        print(f"Error with {sampling_config['name']}: {str(e)}")
        results.append({
            'Sampling Method': sampling_config['name'],
            'Accuracy': 0,
            'Precision': 0,
            'Recall': 0,
            'F1-Score': 0,
            'AUC': 0,
            'Train Size': 0,
            'Test Size': 0,
            'Error': str(e)
        })

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(results_df[['Sampling Method', 'F1-Score', 'Accuracy', 'Precision', 'Recall', 'AUC']].to_string(index=False))

# Sort by F1-Score to see best performing method
print("\n" + "=" * 60)
print("RESULTS SORTED BY F1-SCORE")
print("=" * 60)
sorted_results = results_df.sort_values('F1-Score', ascending=False)
print(sorted_results[['Sampling Method', 'F1-Score', 'Accuracy', 'Precision', 'Recall', 'AUC']].to_string(index=False))

# Save results to CSV
results_df.to_csv('sampling_comparison_results.csv', index=False)
print(f"\nResults saved to: sampling_comparison_results.csv")
