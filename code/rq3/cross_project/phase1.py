import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel
import torch

data = pd.read_excel(r'phase1.xlsx')

numeric_features = data[['Proj_Id', 'Proj_Open', 'CurPriority', 'Sum_Len', 'Desc_Len',
                         'Rel_Num', 'Rel_PCNum', 'Rel_PCPercent', 'Rel_PAve', 'Rel_PMed',
                         'Rep_Num', 'Rep_PCNum', 'Rep_PCPercent', 'Rep_PAve', 'Rep_PMed']].values

labels = data['Changed']
proj_ids = data['Proj_Id'].unique()

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base')


def get_roberta_features(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()


results = []
for proj_id in proj_ids:
    test_data = data[data['Proj_Id'] == proj_id]
    train_data = data[data['Proj_Id'] != proj_id]

    X_train_numeric = train_data[['Proj_Id', 'Proj_Open', 'CurPriority', 'Sum_Len', 'Desc_Len',
                         'Rel_Num', 'Rel_PCNum', 'Rel_PCPercent', 'Rel_PAve', 'Rel_PMed',
                         'Rep_Num', 'Rep_PCNum', 'Rep_PCPercent', 'Rep_PAve', 'Rep_PMed']].values
    y_train = train_data['Changed']

    X_test_numeric = test_data[['Proj_Id', 'Proj_Open', 'CurPriority', 'Sum_Len', 'Desc_Len',
                         'Rel_Num', 'Rel_PCNum', 'Rel_PCPercent', 'Rel_PAve', 'Rel_PMed',
                         'Rep_Num', 'Rep_PCNum', 'Rep_PCPercent', 'Rep_PAve', 'Rep_PMed']].values
    y_test = test_data['Changed']

    if (y_train == 0).sum() > 0 and (y_train == 1).sum() > 0:
        kmeans = KMeans(n_clusters=(y_train == 1).sum())
        kmeans.fit(X_train_numeric[y_train == 0])
        cluster_centers = kmeans.cluster_centers_
        distances = [np.linalg.norm(X_train_numeric[y_train == 0][kmeans.labels_ == i] - center, axis=1) for i, center
                     in enumerate(cluster_centers)]
        nearest_indices = [np.argsort(dist)[:1] for dist in distances]
        selected_samples = np.concatenate(
            [X_train_numeric[y_train == 0][kmeans.labels_ == i][idx] for i, idx in enumerate(nearest_indices)])
        minority_samples = X_train_numeric[y_train == 1]
        combined_samples = np.vstack((selected_samples, minority_samples))
        combined_labels = np.concatenate((np.zeros(len(selected_samples)), np.ones(len(minority_samples))))
    else:
        combined_samples = X_train_numeric
        combined_labels = y_train

    X_train_text = get_roberta_features(train_data['Sum_Content'])
    X_test_text = get_roberta_features(test_data['Sum_Content'])

    X_train_combined = np.hstack((X_train_numeric, X_train_text))
    X_test_combined = np.hstack((X_test_numeric, X_test_text))

    rf_clf = RandomForestClassifier(max_depth=10, n_estimators=280, random_state=42, min_samples_split=5,
                                    min_samples_leaf=2)
    knn_clf = KNeighborsClassifier(n_neighbors=385, metric='manhattan', weights='distance')
    svc_clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=9, learning_rate=0.01,
                            n_estimators=95, subsample=1.0, colsample_bytree=0.5, min_child_weight=1)

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

    voting_clf.fit(X_train_combined, y_train)

    y_pred = voting_clf.predict(X_test_combined)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, voting_clf.predict_proba(X_test_combined)[:, 1])

    results.append({
        'Project_ID': proj_id,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC': auc
    })

results_df = pd.DataFrame(results)
print(results_df)