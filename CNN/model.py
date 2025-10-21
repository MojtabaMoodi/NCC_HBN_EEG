import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGGenderCNN(nn.Module):
    def __init__(self, num_channels=64, num_classes=2):
        super(EEGGenderCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(num_channels, 3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 3))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1, 3))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(1, 3))
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(1, 3))
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(1, 3))
        self.conv8 = nn.Conv2d(256, 256, kernel_size=(1, 3))

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: (batch, 60, 200) -> need to add channel dimension
        if x.dim() == 3:  # (batch, 60, 200)
            x = x.unsqueeze(1)  # (batch, 1, 60, 200)
        x = F.layer_norm(x, x.shape)
        x = F.relu(self.conv1(x))        # After conv1: (batch, 16, 1, 198)
        x = F.relu(self.conv2(x))        # After conv2: (batch, 32, 1, 196)
        x = F.relu(self.conv3(x))        # After conv3: (batch, 64, 1, 194)
        x = F.relu(self.conv4(x))        # After conv4: (batch, 128, 1, 192)
        x = F.relu(self.conv5(x))        # After conv5: (batch, 128, 1, 190)
        x = F.relu(self.conv6(x))        # After conv6: (batch, 256, 1, 188)
        x = F.relu(self.conv7(x))        # After conv7: (batch, 256, 1, 186)
        x = F.relu(self.conv8(x))        # After conv8: (batch, 256, 1, 184)
        
        # Global average pooling across spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)             # (batch, 256)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x