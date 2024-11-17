import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def load_model_for_inference(path, num_classes=10):
    model = MNISTResNet50(num_classes=num_classes).to(device)

    # Load the checkpoint
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path} and ready for inference.")
    return model

def infer(model, inputs):
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        outputs = model(inputs.to(device))
        predictions = torch.argmax(outputs, dim=1)
    return predictions

# Preprocess function for image
def preprocess_image(image_path, input_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(input_size),               # Resize image to match model input
        transforms.ToTensor(),                       # Convert to PyTorch tensor
        transforms.Lambda(lambda x: 1 - x),          # Invert the image
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [-1, 1]
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension


if __name__=="__main__":
    # Load Model
    loaded_model = load_model_for_inference('./checkpoints/ResNet-50_1731794970.374347')

    inference_data_dir = "./Inference_data/" 

    # Process and predict for each image in the folder
    for filename in os.listdir(inference_data_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(inference_data_dir, filename)
            image_tensor = preprocess_image(image_path)
            torch.set_printoptions(threshold=10000)
            # print(image_tensor)
            prediction = infer(loaded_model, image_tensor)
            print(f"Image: {filename}, Predicted Class: {prediction}")