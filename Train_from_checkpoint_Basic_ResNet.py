import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Define ResNet-50 model
class MNISTResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTResNet50, self).__init__()
        # Load pre-trained ResNet-50
        self.resnet = models.resnet50(pretrained=True)
        # Modify the first convolutional layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the fully connected layer with one for MNIST
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint load function
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from {path}, starting from epoch {epoch}")
    return epoch

model = MNISTResNet50(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model = load_checkpoint('./checkpoints/ResNet-50_1731794970.374347', model, optimizer)