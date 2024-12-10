import torch
import torchvision.transforms as transforms
from PIL import Image

def load_model(model_path):
    """
    Load a trained model from the specified path.
    Args:
        model_path (str): Path to the saved model file.
    Returns:
        model (nn.Module): Loaded PyTorch model.
    """
    model = torch.load(model_path)  # Load the model
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image(image_path, transform):
    """
    Preprocess an image for model inference.
    Args:
        image_path (str): Path to the input image.
        transform (torchvision.transforms): Transformations to apply to the image.
    Returns:
        tensor (torch.Tensor): Preprocessed image tensor.
    """
    image = Image.open(image_path).convert('RGB')  # Open the image
    tensor = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension
    return tensor

def predict(model, image_tensor, class_names):
    """
    Perform inference using a trained model.
    Args:
        model (nn.Module): Trained PyTorch model.
        image_tensor (torch.Tensor): Preprocessed input tensor.
        class_names (list): List of class names.
    Returns:
        str: Predicted class name.
    """
    with torch.no_grad():  # No gradients required for inference
        output = model(image_tensor)  # Forward pass
        _, predicted = torch.max(output, 1)  # Get the class index with highest score
        return class_names[predicted.item()]  # Map to class name

# Example usage:
if __name__ == "__main__":
    # Define the class names (modify as per your dataset)
    class_names = ['cat', 'dog', 'car', 'plane']

    # Define preprocessing transformations
