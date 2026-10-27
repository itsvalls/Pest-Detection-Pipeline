import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from datetime import datetime
from xai_utils import GradCAMPlusPlus, overlay_heatmap


# Assuming config.py and models.py are in the same directory or accessible
import config
from models import ModelBuilder

# --- Configuration and Setup ---
PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = config.EXPERIMENTS_DIR

def find_latest_experiment_dir():
    latest_dir = None
    latest_timestamp = None

    if not EXPERIMENTS_DIR.exists():
        return None

    for experiment_dir in EXPERIMENTS_DIR.iterdir():
        if experiment_dir.is_dir() and experiment_dir.name.startswith("run_"):
            try:
                # Extract timestamp from directory name, e.g., "run_20231027_103000"
                timestamp_str = experiment_dir.name.replace("run_", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_dir = experiment_dir
            except ValueError:
                # Ignore directories that don't match the expected timestamp format
                continue
    return latest_dir

latest_experiment_dir = find_latest_experiment_dir()

# Fallback to default models directory if no experiments found
if latest_experiment_dir is None:
    MODELS_DIR = PROJECT_ROOT / "models"
else:
    # MODELS_DIR = latest_experiment_dir / "models"
    MODELS_DIR = Path("experiments\run_20260129_073520\models")

MODEL_ARCHITECTURE = config.MODEL_CONFIG["architecture"]
DROPOUT_RATE = config.MODEL_CONFIG["dropout_rate"]
PRETRAINED = config.MODEL_CONFIG["pretrained"]

IMAGE_SIZE = config.DATASET_CONFIG["image_size"]
NORMALIZE_MEAN = config.DATASET_CONFIG["normalize_mean"]
NORMALIZE_STD = config.DATASET_CONFIG["normalize_std"]

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class to index mapping
class_to_idx_path = MODELS_DIR / "class_indices.json"
if not class_to_idx_path.exists():
    # Attempt to find it in the project root models folder as secondary fallback
    class_to_idx_path = PROJECT_ROOT / "models" / "class_indices.json"
    if not class_to_idx_path.exists():
        print(f"Warning: class_indices.json not found. Using fallback class names.")
        class_to_idx = {
    "aphid": 0,
    "cotton_leaf_roller": 1,
    "dusky_cotton_bug": 2,
    "fall_army_worm": 3,
    "hairy_caterpiller": 4,
    "healthy": 5,
    "jassid": 6,
    "mealybug": 7,
    "pink_bollworm": 8,
    "pod_borer": 9,
    "red_cotton_bug": 10,
    "thrips": 11,
    "white_fly": 12
}
    else:
        with open(class_to_idx_path, "r") as f:
            class_to_idx = json.load(f)
else:
    with open(class_to_idx_path, "r") as f:
        class_to_idx = json.load(f)

idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
NUM_CLASSES = len(class_to_idx)

# --- Model Loading ---
def load_model():
    model = ModelBuilder.build_model(
        num_classes=NUM_CLASSES,
        architecture=MODEL_ARCHITECTURE,
        pretrained=PRETRAINED,
        dropout_rate=DROPOUT_RATE
    )

    # Load the best model checkpoint
    model_path = MODELS_DIR / "best_model.pth" 
    if not model_path.exists():
        # Secondary fallback
        model_path = PROJECT_ROOT / "experiments"  /"run_20260129_073520" /"models" / "best_model.pth"
        print("model path:", model_path)
        if not model_path.exists():
             raise FileNotFoundError(f"Model checkpoint not found. Please train the model first.")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print(f"Model {MODEL_ARCHITECTURE} loaded successfully from {model_path} and moved to {DEVICE}.")
    return model

model = load_model()

# Target layer for ResNet50 is usually layer4[-1]
target_layer = model.layer4[-1]
gradcam_pp = GradCAMPlusPlus(model, target_layer)

# --- Image Preprocessing ---
preprocess_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# --- Prediction Function ---
def predict_image(image: Image.Image):
    if image is None:
        return None, "No image uploaded."

    # Preprocess
    image_tensor = preprocess_transform(image.convert('RGB')).unsqueeze(0).to(DEVICE)

    # Forward pass (need gradients for Grad-CAM++, but we can still get probs first)
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    predicted_idx = probabilities.argmax(dim=1).item()
    top_prob = probabilities[0, predicted_idx].item()
    
    # Thresholding logic
    THRESHOLD = 0.7
    if top_prob < THRESHOLD:
        predicted_class = "Unknown"
    else:
        predicted_class = idx_to_class[predicted_idx]

    # ---- Grad-CAM++ (uses backward pass) ----
    cam = gradcam_pp(image_tensor, class_idx=predicted_idx)

    if cam is not None:
        cam_image = overlay_heatmap(image, cam)
    else:
        cam_image = image

    # Prepare label output
    predictions = {
        idx_to_class[i]: probabilities[0, i].item()
        for i in range(NUM_CLASSES)
    }

    return cam_image, predictions


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Crop Detection - Grad-CAM++") as iface:
    gr.Markdown("# Crop Detection Model (Grad-CAM++) 🌿🌾")
    gr.Markdown("Upload an image of a crop to see where the model focuses using Grad-CAM++.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Crop Image")
            submit_button = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            cam_output = gr.Image(type="pil", label="Grad-CAM++ Visualization")
            output_label = gr.Label(num_top_classes=5, label="Top Predictions")

    # Optional: Add example images if they exist
    example_list = []
    # Try to find some images in the dataset to use as examples
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        # Just grab the first few images found
        count = 0
        for img_path in data_dir.rglob("*.jpg"):
            example_list.append([str(img_path)])
            count += 1
            if count >= 2: break

    if example_list:
        gr.Examples(
            examples=example_list,
            inputs=image_input
        )

    submit_button.click(
        fn=predict_image, 
        inputs=image_input, 
        outputs=[cam_output, output_label]
    )

if __name__ == "__main__":
    iface.launch(share=True)
