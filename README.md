# Pest Detection Pipeline

A 13-class **ResNet-50** pest classifier for multi-crop pest identification, with **Grad-CAM++** explainability and a Gradio demo app. Developed under the **TIH-IoT CHANAKYA Fellowship 2026** (Technology Innovation Hub for IoT and IoE, IIT Bombay) and deployed in the **Mahavistaar** farmer-advisory application.

📄 **Full report:** [Pest_Detection_Report.pdf](Pest_Detection_Report.pdf)

## Dataset

The pest images (the **PoCRA pest dataset**, 30,226 labelled images, 13 classes) were **obtained from TIH-IoT, IIT Bombay**. The dataset is not redistributed in this repository; please request access from TIH-IoT, IIT Bombay.

Classes: Aphid, Cotton Leaf Roller, Dusky Cotton Bug, Fall Army Worm, Hairy Caterpillar, Healthy, Jassid, Mealybug, Pink Bollworm, Pod Borer, Red Cotton Bug, Thrips, White Fly.

| Split | Images | Share |
|---|---|---|
| Training | 21,168 | 70.0% |
| Validation | 4,520 | 15.0% |
| Test | 4,538 | 15.0% |
| **Total** | **30,226** | 100% |

## Method

- **Backbone:** ResNet-50 (ImageNet-pretrained), 23,534,669 parameters. Head: Dropout(0.3) + Linear(2048 → 13).
- **Two-phase transfer learning**
  - Phase 1: backbone frozen, only the head trained (26,637 params, 0.11%).
  - Phase 2: `layer3` and `layer4` unfrozen (~22.1M params, ~94%).
- **Loss:** cross-entropy with inverse-frequency class weights (0.38 for Jassid up to 4.47 for Cotton Leaf Roller).
- **Optimiser:** Adam, lr 0.001, weight decay 0.0001; batch size 32; up to 50 epochs per phase; early stopping (patience 5, min delta 0.0001).
- **Explainability:** Grad-CAM++ on the last residual block (`layer4[-1]`); confidence threshold 0.7 (below it, the app returns "Unknown").

## Results (held-out test set, 4,538 images)

| Metric | Value |
|---|---|
| Accuracy | **98.3%** |
| Weighted precision / recall / F1 | **0.983 / 0.983 / 0.983** |
| Macro-F1 | ≈ 0.985 |
| Worst class F1 | 0.960 (Thrips) |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Aphid | 0.987 | 0.995 | 0.991 | 376 |
| Cotton Leaf Roller | 1.000 | 0.987 | 0.994 | 78 |
| Dusky Cotton Bug | 0.993 | 1.000 | 0.997 | 149 |
| Fall Army Worm | 0.984 | 0.984 | 0.984 | 188 |
| Hairy Caterpillar | 0.983 | 0.975 | 0.979 | 121 |
| Healthy | 0.969 | 0.976 | 0.972 | 287 |
| Jassid | 0.978 | 0.975 | 0.977 | 921 |
| Mealybug | 1.000 | 1.000 | 1.000 | 193 |
| Pink Bollworm | 0.991 | 0.994 | 0.993 | 338 |
| Pod Borer | 0.991 | 0.988 | 0.989 | 894 |
| Red Cotton Bug | 1.000 | 0.993 | 0.996 | 142 |
| Thrips | 0.954 | 0.966 | 0.960 | 325 |
| White Fly | 0.981 | 0.977 | 0.979 | 526 |
| **Weighted avg** | **0.983** | **0.983** | **0.983** | **4,538** |

## Repository structure

| File | Purpose |
|---|---|
| `models.py` | `ModelBuilder`: builds ResNet-50 / ResNet-18 / MobileNetV2 / EfficientNet-B0 with a custom head; backbone freezing helpers |
| `training.py` | `ModelTrainer` (training loop, class-weighted loss, schedulers, checkpointing) and `EarlyStopping` |
| `evaluation.py` | `ModelEvaluator`: accuracy/precision/recall/F1, confusion matrix, classification report, saved results |
| `xai_utils.py` | `GradCAM`, `GradCAMPlusPlus` and `overlay_heatmap` |
| `app_gradcampp.py` | Gradio web app: upload an image, see class probabilities and the Grad-CAM++ overlay |
| `Pest_Detection_Report.pdf` | Full project report |

## Setup

```bash
git clone https://github.com/<your-username>/Pest-Detection-Pipeline.git
cd Pest-Detection-Pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is recommended for training; inference runs on CPU too.

## Configuration

The modules read settings from a `config.py` module (imported as `import config`). It holds the values listed below; add it next to the scripts and adjust to your paths:

```python
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments")
RESULTS_DIR = Path("results")

MODEL_CONFIG = {"architecture": "resnet50", "dropout_rate": 0.3, "pretrained": True}
DATASET_CONFIG = {
    "image_size": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
}
TRAINING_CONFIG = {
    "optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 1e-4,
    "scheduler": "cosine", "num_epochs": 50,
    "early_stopping_patience": 5, "early_stopping_min_delta": 1e-4,
}
CHECKPOINT_CONFIG = {
    "best_model_path": Path("models/best_model.pth"),
    "final_model_path": Path("models/final_model.pth"),
}
```

Place the dataset under `data/<class_name>/*.jpg` (one folder per class) and create the 70/15/15 split with your own data loaders.

## Train

```python
import config
from models import ModelBuilder
from training import ModelTrainer

model = ModelBuilder.build_model(num_classes=13, architecture="resnet50",
                                 pretrained=True, dropout_rate=0.3)

# Phase 1: train the head only
ModelBuilder.freeze_backbone(model)
trainer = ModelTrainer(model, config, class_weights=class_weights)  # tensor of 13 weights
trainer.train(train_loader, val_loader)

# Phase 2: unfreeze layer3 + layer4 and fine-tune
for name, p in model.named_parameters():
    p.requires_grad = name.startswith(("layer3", "layer4", "fc"))
trainer = ModelTrainer(model, config, class_weights=class_weights)
trainer.train(train_loader, val_loader)
```

## Evaluate

```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator(model, config, device="cuda")
metrics, preds, labels, probs = evaluator.evaluate(test_loader, class_to_idx)
evaluator.print_metrics(metrics)
evaluator.save_results(metrics, preds, labels, class_to_idx)  # metrics.json, predictions.csv, confusion_matrix.png
```

## Run the demo app (Grad-CAM++)

```bash
python app_gradcampp.py
```

The app loads `best_model.pth` and `class_indices.json` from the experiment's `models/` folder. Update the hardcoded `experiments/run_<timestamp>/models` path in `app_gradcampp.py` to point at your trained run. It opens a local Gradio URL (and a public share link, since `share=True`).

## Deployment

The model serves behind a prediction API (request validation → preprocessing → crop check → classification → confidence filter → structured response) and is live in the **Mahavistaar** farmer-advisory application. It was showcased at the AI Summit under Hon. Chief Minister of Maharashtra Devendra Fadnavis.

## Acknowledgements

- Pest dataset: **TIH-IoT, IIT Bombay** (PoCRA pest dataset)
- **TIH-IoT CHANAKYA Fellowship 2026**
- NMIMS MPSTME
