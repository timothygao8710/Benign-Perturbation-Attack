from data_loader import prepare_data
from MLP import BinaryClassifier, train_model, evaluate_model
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Config:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 8
        self.num_epochs = 1000
        # self.patience = 2000
        self.hidden_sizes = [32, 32]
        self.dropout_rate = 0
        self.test_size = 0.1
        self.random_seed = 420
        self.use_batch_norm = False

def main():
    config = Config()
    
    # Load and process data
    file_path = '/accounts/projects/binyu/timothygao/Benign-Perturbation-Attack/data/edit_distance_v_accuracy'
    train_loader, test_loader, input_shape = prepare_data(file_path, config, scale=False)
    
    # Create and train model
    model = BinaryClassifier(input_shape, config.hidden_sizes, config.dropout_rate, config.use_batch_norm)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, config)
    
    # Evaluate model
    evaluate_model(trained_model, test_loader)

if __name__ == "__main__":
    main()