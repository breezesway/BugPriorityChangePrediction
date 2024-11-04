import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

majority_indices = np.where(labels == 0)[0]
minority_indices = np.where(labels == 1)[0]

np.random.seed(42)
selected_majority_indices = np.random.choice(majority_indices, size=len(minority_indices), replace=False)

selected_samples = np.vstack((numeric_features[selected_majority_indices], numeric_features[minority_indices]))
combined_labels = np.concatenate((np.zeros(len(selected_majority_indices)), np.ones(len(minority_indices))))

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base')

def get_roberta_features(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

selected_texts = data.loc[selected_majority_indices]['Sum_Content']
minority_texts = data.loc[minority_indices]['Sum_Content']
combined_texts = pd.concat([selected_texts, minority_texts])

text_features = get_roberta_features(combined_texts.tolist())

combined_features = np.hstack((selected_samples, text_features))

X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_labels, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(max_depth=10, n_estimators=280, random_state=42, min_samples_split=5, min_samples_leaf=2)
knn_clf = KNeighborsClassifier(n_neighbors=385, metric='manhattan', weights='distance')
svc_clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=9, learning_rate=0.01, n_estimators=95, subsample=1.0, colsample_bytree=0.5, min_child_weight=1)

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

voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1]))