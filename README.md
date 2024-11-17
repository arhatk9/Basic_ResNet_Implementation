# Basic_ResNet_Implementation

This project implements a **ResNet-50** model for training and testing on the **MNIST dataset**. It also includes functionality for saving checkpoints during training and using the trained model for inference on custom data.

---

## Features

1. **Model Definition:**
   - A modified ResNet-50 to handle MNIST's grayscale images (1 channel).
   - Outputs predictions for 10 classes (digits 0-9).

2. **Training:**
   - Trains the model on the MNIST dataset.
   - Saves the model state and optimizer state as checkpoints.

3. **Testing:**
   - Evaluates the model on the MNIST test set.

4. **Checkpointing:**
   - Saves model state, optimizer state, and epoch information.
   - Allows resuming training or using the model for inference.

5. **Inference:**
   - Predicts classes for custom grayscale images using the trained model.

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow (for image handling)

Install dependencies with:
```bash
pip install torch torchvision pillow
