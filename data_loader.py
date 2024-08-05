import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded data from pickle file: {file_path}")
    return data

def process_data(data):
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        print("Data is not in the expected format (list of dictionaries).")
        return None, None

    df = pd.DataFrame(data)
    
    # print(df.head())
    
    if 'is_correct' not in df.columns:
        print("Data is missing the 'is_correct' column.")
        return None, None

    # Use all columns except 'is_correct' as features
    feature_columns = [col for col in df.columns if col != 'is_correct']
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    features = df[feature_columns].copy()
    output = df['is_correct'].copy()

    features = features.select_dtypes(include=[np.number])
    
    for col in feature_columns:
        if col not in features.columns:
            print(f"Column '{col}' was dropped because it is not numeric.")
        
    output = output.astype(int)
    
    print(f"Total number of rows after processing: {len(features)}")
    print(f"Feature columns: {features.columns}")
    return features, output

class BinaryClassificationDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features.iloc[idx].values
        label = float(self.labels.iloc[idx])
        
        if self.transform:
            features = self.transform(features)
        
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

def prepare_data(file_path, config, scale):
    raw_data = load_data(file_path)
    features, output = process_data(raw_data)
    
    if features is None or output is None:
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=config.test_size, random_state=config.random_seed)
    
    scaler = FeatureScaler() if scale else None
    train_dataset = BinaryClassificationDataset(X_train, y_train, transform=scaler)
    test_dataset = BinaryClassificationDataset(X_test, y_test, transform=scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    return train_loader, test_loader, X_train.shape[1]

def main():
    class MockConfig:
        def __init__(self):
            self.test_size = 0.2
            self.random_seed = 42
            self.batch_size = 32

    config = MockConfig()
    file_path = '/accounts/projects/binyu/timothygao/Benign-Perturbation-Attack/data/medqa_llama'  # Adjust this path as necessary
    
    try:
        raw_data = load_data(file_path)
        print(f"Number of items in raw data: {len(raw_data)}")
        print("First item in raw data:")
        print(raw_data[0])

        features, output = process_data(raw_data)
        
        if features is not None and output is not None:
            print("\nFeatures info:")
            print(features.info())
            print("\nOutput info:")
            print(output.value_counts())

            train_loader, test_loader, input_shape = prepare_data(file_path, config)
            
            if train_loader and test_loader:
                print(f"\nInput shape: {input_shape}")
                print(f"Number of batches in train_loader: {len(train_loader)}")
                print(f"Number of batches in test_loader: {len(test_loader)}")
                
                # Test a couple of samples from the train_loader
                for i, (batch_features, batch_labels) in enumerate(train_loader):
                    if i >= 2:  # Only test the first two batches
                        break
                    print(f"\nBatch {i+1}:")
                    print(f"Features shape: {batch_features.shape}")
                    print(f"Labels shape: {batch_labels.shape}")
                    print(f"First few features:\n{batch_features[:5]}")
                    print(f"First few labels: {batch_labels[:5]}")
            else:
                print("Failed to create data loaders.")
        else:
            print("Failed to process data.")
    
    except Exception as e:
        print(f"An error occurred during data loading and processing: {str(e)}")

if __name__ == "__main__":
    main()