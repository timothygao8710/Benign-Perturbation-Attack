import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *


class Config:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 8
        self.num_epochs = 1000
        self.patience = 2000
        self.hidden_sizes = [32, 32]
        self.dropout_rate = 0
        self.test_size = 0.2
        self.random_seed = 420
        self.use_batch_norm = False


def process_data(data):
    features, output = [], []
    for row in data:
        
        output.append(row["is_correct"])
        cur_features = []

        all_features = []
        for i in row.keys():
            if i == "is_correct" or not isinstance(row[i], (int, float)):
                continue
            
            # if i not in ["peturbed_entropy_20", "peturbed_entropy_15", "peturbed_entropy_10", "peturbed_entropy_5"]:
            #     continue
            
            # if i not in ["original_entropy"]:
            #     continue
            
            # if "correct" in i or "row" in i:
            #     continue
            
            
            # if "sensitivity" not in i:
            #     continue
            
            cur_features.append(row[i])
            all_features.append(i)

        print(f"all features {all_features}")
        
        # cur_features.extend(row["model_prob"])
        # diffs = [row["model_prob"][i-1] - row["model_prob"][i] for i in range(1, len(row["model_prob"]))]
        # cur_features.extend(diffs)
        
        features.append(cur_features)

    print(f"Total number of rows {len(data)}")
    return np.array(features), np.array(output)


class BinaryClassificationDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = float(self.labels[idx])
        
        if self.transform:
            features = self.transform(features)
        
        # print(features, label)
        return torch.FloatTensor(features), torch.FloatTensor([label])
    
class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fit = False

    def __call__(self, features):
        features_np = np.array(features).reshape(1, -1)
        if not self.is_fit:
            features_np = self.scaler.fit_transform(features_np)
            self.is_fit = True
        else:
            features_np = self.scaler.transform(features_np)
        return features_np.squeeze()

# class BinaryClassifier(nn.Module):
#     def __init__(self, input_shape, hidden_sizes, dropout_rate, use_batch_norm):
#         super(BinaryClassifier, self).__init__()
#         self.layer1 = nn.Linear(input_shape, 64)
#         self.layer2 = nn.Linear(64, 32)
#         self.layer3 = nn.Linear(32, 16)
#         self.layer4 = nn.Linear(16, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         x = self.dropout(self.relu(self.layer1(x)))
#         x = self.dropout(self.relu(self.layer2(x)))
#         x = self.relu(self.layer3(x))
#         x = self.sigmoid(self.layer4(x))
#         return x

class BinaryClassifier(nn.Module):
    def __init__(self, input_shape, hidden_sizes, dropout_rate, use_batch_norm):
        super(BinaryClassifier, self).__init__()
        layers = []
        prev_size = input_shape
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(config.num_epochs):

        
        model.train()
        train_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        print(f'Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        scheduler.step(val_loss)
        
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_no_improve = 0
        #     torch.save(model.state_dict(), 'best_model.pth')
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve == config.patience:
        #         print('Early stopping!')
        #         model.load_state_dict(torch.load('best_model.pth'))
        #         return model
    
    return model

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

def main():
    config = Config()
    
    # Load and process data
    file_path = './data/edit_distance_v_accuracy_K_10'
    raw_data = load_data(file_path)
    features, output = process_data(raw_data)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=config.test_size, random_state=config.random_seed)
    
    # Create datasets and dataloaders
    # scaler = FeatureScaler()
    scaler = None
    train_dataset = BinaryClassificationDataset(X_train, y_train, transform=scaler)
    test_dataset = BinaryClassificationDataset(X_test, y_test, transform=scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Get input shape from the first item in the dataset
    input_shape = X_train.shape[1]
        
    # Create and train model
    model = BinaryClassifier(input_shape, config.hidden_sizes, config.dropout_rate, config.use_batch_norm)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config)
    
    # Evaluate model
    evaluate_model(trained_model, test_loader)

if __name__ == "__main__":
    main()