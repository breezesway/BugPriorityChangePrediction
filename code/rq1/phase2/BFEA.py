import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

df = pd.read_excel('phase_2.xlsx')

columns_to_drop = ['key', 'curTime', 'Project', 'Rel_Labels',
                   'Rel_Versions', 'Rel_FixVersions', 'Rel_Components',
                   'Rel_Attachments', 'Rep', 'Agn']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

X_resampled = df.drop(columns=['TargetPriority'])
y_resampled = df['TargetPriority']

class_weights = 1.0 / (y_resampled.value_counts(normalize=True) * len(y_resampled.unique()))
class_weights = class_weights.sort_index().values

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')


def get_roberta_features(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # 取<s>位置的特征向量


X_resampled['combined_text'] = X_resampled['Sum_Content'] + ' ' + X_resampled['Cmt_Content']
text_features = get_roberta_features(X_resampled['combined_text'].tolist())

other_features = X_resampled.drop(['combined_text', 'Sum_Content', 'Cmt_Content'], axis=1).values
final_features = np.hstack((text_features, other_features))


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_features, test_features, train_labels, test_labels = train_test_split(final_features, y_resampled, test_size=0.2,
                                                                            random_state=42)

train_dataset = CustomDataset(train_features, train_labels)
test_dataset = CustomDataset(test_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


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


input_dim = final_features.shape[1]
net = Net(input_dim)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 86
for epoch in range(num_epochs):
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
print(f'F1-weighted score: {f1_weighted}')
print(f'F1-macro score: {f1_macro}')