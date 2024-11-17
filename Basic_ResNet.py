import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import datetime

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ResNet-50 input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

model = MNISTResNet50(num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Checkpoint save function
def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Save checkpoint after every epoch if epoch==10
        if epoch==num_epochs-1:
            save_checkpoint(model, optimizer, epoch, path=f'./checkpoints/ResNet-50_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

# Testing function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Train and test the model
train_model(model, train_loader, criterion, optimizer, num_epochs=1)
test_model(model, test_loader)

# Save predictions for a few test samples
samples, labels = next(iter(test_loader))
samples, labels = samples.to(device), labels.to(device)
outputs = model(samples)
_, predicted = torch.max(outputs, 1)

print("Predicted:", predicted[:5].cpu().numpy())
print("True Labels:", labels[:5].cpu().numpy())
