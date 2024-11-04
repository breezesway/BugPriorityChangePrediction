import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

df = pd.read_excel('phase2.xlsx')

columns_to_drop = ['key', 'curTime', 'Project', 'Rel_Labels',
                   'Rel_Versions', 'Rel_FixVersions', 'Rel_Components',
                   'Rel_Attachments', 'Rep', 'Agn']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

proj_ids = df['Proj_Id'].unique()

def custom_sampling(df):
    smote = SMOTE(sampling_strategy={1: int(len(df[df['curPriority'] == 'Minor']) * 1.65),
                                     2: int(len(df[df['curPriority'] == 'Blocker']) * 1.15),
                                     3: int(len(df[df['curPriority'] == 'Critical']) * 1.15),
                                     4: int(len(df[df['curPriority'] == 'Trivial']) * 1.8)})
    rus = RandomUnderSampler(sampling_strategy={5: int(len(df[df['curPriority'] == 'Major']) * 0.9)})

    df_minor = df[df['curPriority'] == 'Minor']
    df_blocker = df[df['curPriority'] == 'Blocker']
    df_critical = df[df['curPriority'] == 'Critical']
    df_major = df[df['curPriority'] == 'Major']
    df_trivial = df[df['curPriority'] == 'Trivial']

    X_minor, y_minor = smote.fit_resample(df_minor.drop('TargetPriority', axis=1), df_minor['TargetPriority'])
    X_blocker, y_blocker = smote.fit_resample(df_blocker.drop('TargetPriority', axis=1), df_blocker['TargetPriority'])
    X_critical, y_critical = smote.fit_resample(df_critical.drop('TargetPriority', axis=1), df_critical['TargetPriority'])
    X_major, y_major = rus.fit_resample(df_major.drop('TargetPriority', axis=1), df_major['TargetPriority'])
    X_trivial, y_trivial = smote.fit_resample(df_trivial.drop('TargetPriority', axis=1), df_trivial['TargetPriority'])

    X_resampled = pd.concat([X_minor, X_blocker, X_critical, X_major, X_trivial])
    y_resampled = pd.concat([y_minor, y_blocker, y_critical, y_major, y_trivial])

    return X_resampled, y_resampled

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

def get_roberta_features(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # 取<s>位置的特征向量

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

results = []

for proj_id in proj_ids:
    test_data = df[df['Proj_Id'] == proj_id]
    train_data = df[df['Proj_Id'] != proj_id]

    X_train_resampled, y_train_resampled = custom_sampling(train_data)

    class_weights = 1.0 / (y_train_resampled.value_counts(normalize=True) * len(y_train_resampled.unique()))
    class_weights = class_weights.sort_index().values

    X_train_resampled['combined_text'] = X_train_resampled['Sum_Content'] + ' ' + X_train_resampled['Cmt_Content']
    X_test_resampled = test_data.copy()
    X_test_resampled['combined_text'] = X_test_resampled['Sum_Content'] + ' ' + X_test_resampled['Cmt_Content']

    train_text_features = get_roberta_features(X_train_resampled['combined_text'].tolist())
    test_text_features = get_roberta_features(X_test_resampled['combined_text'].tolist())

    X_train_numeric = X_train_resampled.drop(['combined_text', 'Sum_Content', 'Cmt_Content', 'TargetPriority'], axis=1).values
    X_test_numeric = X_test_resampled.drop(['combined_text', 'Sum_Content', 'Cmt_Content', 'TargetPriority'], axis=1).values

    X_train_final = np.hstack((train_text_features, X_train_numeric))
    X_test_final = np.hstack((test_text_features, X_test_numeric))

    train_dataset = CustomDataset(X_train_final, y_train_resampled)
    test_dataset = CustomDataset(X_test_final, test_data['TargetPriority'])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_dim = X_train_final.shape[1]
    net = Net(input_dim)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 86
    for epoch in range(num_epochs):
        net.train()
        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def evaluate(model, dataloader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in dataloader:
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        return all_preds, all_labels

    preds, labels = evaluate(net, test_loader)
    f1_weighted = f1_score(labels, preds, average='weighted')
    f1_macro = f1_score(labels, preds, average='macro')

    results.append({
        'Project_ID': proj_id,
        'F1-Weighted': f1_weighted,
        'F1-Macro': f1_macro
    })

results_df = pd.DataFrame(results)
print(results_df)