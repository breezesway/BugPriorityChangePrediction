import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_excel('../../data/phase_2.xlsx')

# Drop unnecessary columns
columns_to_drop = ['key', 'curTime', 'Project', 'Rel_Labels',
                   'Rel_Versions', 'Rel_FixVersions', 'Rel_Components',
                   'Rel_Attachments', 'Rep', 'Agn']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Load RoBERTa model for text features
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

def get_roberta_features(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def prepare_features(df):
    """Prepare features combining text and other features"""
    df_copy = df.copy()
    df_copy['combined_text'] = df_copy['Sum_Content'] + ' ' + df_copy['Cmt_Content']
    text_features = get_roberta_features(df_copy['combined_text'].tolist())
    
    other_features = df_copy.drop(['combined_text', 'Sum_Content', 'Cmt_Content', 'TargetPriority'], axis=1).values
    final_features = np.hstack((text_features, other_features))
    
    return final_features, df_copy['TargetPriority']

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

def train_model(X_train, y_train, X_test, y_test, num_epochs=50):
    """Train the neural network model"""
    # Calculate class weights
    class_weights = 1.0 / (y_train.value_counts(normalize=True) * len(y_train.unique()))
    class_weights = class_weights.sort_index().values
    
    # Create datasets and dataloaders
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    net = Net(input_dim)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    preds, labels = evaluate(net, test_loader)
    f1_weighted = f1_score(labels, preds, average='weighted')
    f1_macro = f1_score(labels, preds, average='macro')
    
    return f1_weighted, f1_macro

# Define sampling strategies to test
sampling_strategies = [
    {
        'name': 'No sampling',
        'method': None,
        'description': '--'
    },
    {
        'name': 'SMOTE oversampling',
        'method': 'smote_oversampling',
        'description': 'Trivial +100%, Minor +50%',
        'params': {
            'Trivial': 1.0,  # +100%
            'Minor': 0.5      # +50%
        }
    },
    {
        'name': 'Random undersampling -10%',
        'method': 'random_undersampling',
        'description': 'Blocker/Critical/Major -10%',
        'params': {
            'Blocker': 0.9,   # -10%
            'Critical': 0.9,  # -10%
            'Major': 0.9      # -10%
        }
    },
    {
        'name': 'Random undersampling -20%',
        'method': 'random_undersampling',
        'description': 'Blocker/Critical/Major -20%',
        'params': {
            'Blocker': 0.8,   # -20%
            'Critical': 0.8,  # -20%
            'Major': 0.8      # -20%
        }
    },
    {
        'name': 'Random undersampling -30%',
        'method': 'random_undersampling',
        'description': 'Blocker/Critical/Major -30%',
        'params': {
            'Blocker': 0.7,   # -30%
            'Critical': 0.7,  # -30%
            'Major': 0.7      # -30%
        }
    },
    {
        'name': 'Mixed sampling',
        'method': 'mixed_sampling',
        'description': 'Trivial +100%, Minor +50%; Blocker/Critical/Major -20%',
        'params': {
            'oversample': {'Trivial': 1.0, 'Minor': 0.5},
            'undersample': {'Blocker': 0.8, 'Critical': 0.8, 'Major': 0.8}
        }
    }
]

def apply_sampling_strategy(df, strategy):
    """Apply the specified sampling strategy"""
    if strategy['method'] is None:
        # No sampling
        return df
    
    elif strategy['method'] == 'smote_oversampling':
        # SMOTE oversampling for specified classes
        sampling_strategy = {}
        for priority, ratio in strategy['params'].items():
            current_count = len(df[df['curPriority'] == priority])
            target_count = int(current_count * (1 + ratio))
            sampling_strategy[priority] = target_count
        
        smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(df.drop('TargetPriority', axis=1), df['TargetPriority'])
        
        # Reconstruct dataframe
        df_resampled = X_resampled.copy()
        df_resampled['TargetPriority'] = y_resampled
        return df_resampled
    
    elif strategy['method'] == 'random_undersampling':
        # Random undersampling for specified classes
        sampling_strategy = {}
        for priority, ratio in strategy['params'].items():
            current_count = len(df[df['curPriority'] == priority])
            target_count = int(current_count * ratio)
            sampling_strategy[priority] = target_count
        
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(df.drop('TargetPriority', axis=1), df['TargetPriority'])
        
        # Reconstruct dataframe
        df_resampled = X_resampled.copy()
        df_resampled['TargetPriority'] = y_resampled
        return df_resampled
    
    elif strategy['method'] == 'mixed_sampling':
        # Mixed sampling: oversample some classes, undersample others
        # First oversample
        oversample_strategy = {}
        for priority, ratio in strategy['params']['oversample'].items():
            current_count = len(df[df['curPriority'] == priority])
            target_count = int(current_count * (1 + ratio))
            oversample_strategy[priority] = target_count
        
        smote = SMOTE(sampling_strategy=oversample_strategy, k_neighbors=5, random_state=42)
        df_oversampled = df.copy()
        
        # Apply oversampling to each class separately
        df_list = []
        for priority in oversample_strategy.keys():
            df_priority = df[df['curPriority'] == priority]
            if len(df_priority) > 0:
                X_oversampled, y_oversampled = smote.fit_resample(
                    df_priority.drop('TargetPriority', axis=1), 
                    df_priority['TargetPriority']
                )
                df_oversampled_priority = X_oversampled.copy()
                df_oversampled_priority['TargetPriority'] = y_oversampled
                df_list.append(df_oversampled_priority)
        
        # Add non-oversampled classes
        for priority in df['curPriority'].unique():
            if priority not in oversample_strategy:
                df_list.append(df[df['curPriority'] == priority])
        
        df_oversampled = pd.concat(df_list, ignore_index=True)
        
        # Then undersample
        undersample_strategy = {}
        for priority, ratio in strategy['params']['undersample'].items():
            current_count = len(df_oversampled[df_oversampled['curPriority'] == priority])
            target_count = int(current_count * ratio)
            undersample_strategy[priority] = target_count
        
        rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(
            df_oversampled.drop('TargetPriority', axis=1), 
            df_oversampled['TargetPriority']
        )
        
        # Reconstruct dataframe
        df_resampled = X_resampled.copy()
        df_resampled['TargetPriority'] = y_resampled
        return df_resampled

# Results storage
results = []

print("Testing different sampling strategies for Phase II...")
print("=" * 70)

for strategy in sampling_strategies:
    print(f"\nTesting: {strategy['name']}")
    print(f"Strategy: {strategy['description']}")
    
    try:
        # Apply sampling strategy
        df_sampled = apply_sampling_strategy(df, strategy)
        
        # Prepare features
        X_features, y_labels = prepare_features(df_sampled)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        
        # Train model and get results
        f1_weighted, f1_macro = train_model(X_train, y_train, X_test, y_test, num_epochs=50)
        
        # Store results
        results.append({
            'Method': strategy['name'],
            'Strategy': strategy['description'],
            'F1-weighted': f1_weighted,
            'F1-macro': f1_macro,
            'Train Size': len(X_train),
            'Test Size': len(X_test)
        })
        
        print(f"F1-weighted: {f1_weighted:.3f}")
        print(f"F1-macro: {f1_macro:.3f}")
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
    except Exception as e:
        print(f"Error with {strategy['name']}: {str(e)}")
        results.append({
            'Method': strategy['name'],
            'Strategy': strategy['description'],
            'F1-weighted': 0,
            'F1-macro': 0,
            'Train Size': 0,
            'Test Size': 0,
            'Error': str(e)
        })

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(results_df[['Method', 'Strategy', 'F1-weighted', 'F1-macro']].to_string(index=False))

# Sort by F1-weighted to see best performing method
print("\n" + "=" * 70)
print("RESULTS SORTED BY F1-WEIGHTED")
print("=" * 70)
sorted_results = results_df.sort_values('F1-weighted', ascending=False)
print(sorted_results[['Method', 'Strategy', 'F1-weighted', 'F1-macro']].to_string(index=False))

# Save results to CSV
results_df.to_csv('phase2_sampling_comparison_results.csv', index=False)
print(f"\nResults saved to: phase2_sampling_comparison_results.csv")

# Print the best method
best_method = sorted_results.iloc[0]
print(f"\nBest performing method: {best_method['Method']} with F1-weighted: {best_method['F1-weighted']:.3f}")
