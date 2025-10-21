import sys
import torch
import numpy as np
from model import EEGGenderCNN
from train import train_model
from evaluate import evaluate_model

# Add the data_processing directory to the path
sys.path.append('/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing')
from eeg_dataset import EEGDataLoader
from target_transforms import gender_classification_transform


def main():
    # Use the new preprocessed data
    pickle_dir = "/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data"
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create data loaders using new pipeline
    print("Creating data loaders with new pipeline...")
    train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
        pickle_dir=pickle_dir,
        batch_size=32,
        num_workers=4,
        random_seed=42,
        task_type="both",  # Use both active and passive tasks
        target_type="gender",  # Focus on gender classification
        gender_transform=gender_classification_transform,  # Apply gender transform
        age_transform=None,  # No age transform needed for gender classification
        transform=None  # No additional EEG transforms for now
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader)} batches") 
    print(f"Test loader: {len(test_loader)} batches")

    # Train the model
    print("\nStarting model training...")
    model = EEGGenderCNN(num_channels=60)
    print("Beginning training...")
    trained_model = train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001, batch_size=32)
    print("Training completed.")
    
    # Evaluate the model
    evaluate_model("checkpoints/best_model.pth", test_loader)

if __name__ == "__main__":
    main()
