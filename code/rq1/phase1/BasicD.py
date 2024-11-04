import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel
import torch

data = pd.read_excel(r'phase1.xlsx')

numeric_features = data[['Proj_Id', 'CurPriority', 'Sum_Len', 'Desc_Len',
                         'Rel_Num', 'Rel_PCNum', 'Rel_PCPercent', 'Rel_PAve', 'Rel_PMed',
                         'Rep_Num', 'Rep_PCNum', 'Rep_PCPercent']].values

labels = data['Changed']

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

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base')


def get_roberta_features(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()


selected_texts = data.loc[data['Changed'] == 0].iloc[np.concatenate([idx for idx in nearest_indices])]['Sum_Content']
minority_texts = data.loc[data['Changed'] == 1]['Sum_Content']
combined_texts = pd.concat([selected_texts, minority_texts])

text_features = get_roberta_features(combined_texts.tolist())

combined_features = np.hstack((combined_samples, text_features))

X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_labels, test_size=0.2, random_state=42)

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=9, learning_rate=0.01,
                        n_estimators=95, subsample=1.0, colsample_bytree=0.5, min_child_weight=1)

xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1]))
