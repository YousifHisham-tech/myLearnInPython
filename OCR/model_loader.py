
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_production_model(model_save_path, num_classes=2):
    """
    Loads a pre-trained Faster R-CNN model and sets it up for production inference.

    Args:
        model_save_path (str): Path to the saved model checkpoint (.pth file).
        num_classes (int): Number of classes the model was trained on (including background).
        Default is 2 (1 for background + 1 for table).

    Returns:
        tuple: A tuple containing the loaded model and the device it's moved to.
    """

    # 1. Initialize the base model architecture
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    # 2. Adjust the classifier head for your specific number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes  # num_classes: 1 for background + number of actual classes
    )

    # 3. Load the saved state dictionary
    checkpoint = torch.load(model_save_path, map_location=torch.device('cpu')) # Load to CPU first
    model.load_state_dict(checkpoint['model_state_dict'])

    # 4. Determine device and move model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 5. Set model to evaluation mode for inference
    model.eval()

    print(f"Model loaded for production successfully from {model_save_path} on device: {device}")
    return model, device

if __name__ == '__main__':
    # Example usage if this file is run directly
    # In a real scenario, you would typically import and use the function
    # from another script.

    # This path should point to where your .pth file is located
    my_model_path = 'fasterrcnn_table_detector.pth'

    try:
        production_model, inference_device = load_production_model(my_model_path, num_classes=2)
        print("Production model is ready for inference!")
        # You can now use production_model for predictions
        # For example, pass an image through it:
        # dummy_input = torch.randn(1, 3, 800, 800).to(inference_device)
        # with torch.no_grad():
        #     output = production_model(dummy_input)
        # print(output)
    except FileNotFoundError:
        print(f"Error: Model file not found at {my_model_path}")
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
