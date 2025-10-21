import torch
from torch.utils.data import DataLoader
from model import EEGGenderCNN

def evaluate_model(checkpoint_path, test_loader):
    model = EEGGenderCNN(num_channels=60, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['eeg_data'].to(device)
            labels = batch['gender'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test dataset: {accuracy:.2f}%')


