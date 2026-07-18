import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset.config import (
    DATASET_PATH,
    MODEL_DIR,
    GRADCAM_DIR,
    IMG_SIZE,
    CLASS_NAMES,
)
from preprocessing.preprocessing import get_test_transform
from models.transfer import TransferLearningModel
from explainability.gradcam import GradCAM, overlay_gradcam

os.makedirs(GRADCAM_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 50)
print("GRAD-CAM — ResNet18")
print("=" * 50)
print("Dispositivo:", device)

# ── Modelo ────────────────────────────────────────────────────
model = TransferLearningModel().to(device)
model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "best_resnet18.pth"),
               map_location=device)
)
model.eval()

# ── Coleta imagens do dataset ─────────────────────────────────
image_paths = [
    os.path.join(root, f)
    for root, _, files in os.walk(DATASET_PATH)
    for f in files
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
]

if not image_paths:
    raise FileNotFoundError(
        f"Nenhuma imagem encontrada em: {DATASET_PATH}"
    )

# Gera Grad-CAM para N imagens aleatórias (1 benigna + 1 maligna)
selected = []
for cls in CLASS_NAMES:
    candidates = [p for p in image_paths if cls in p.lower()]
    if candidates:
        selected.append(random.choice(candidates))

transform = get_test_transform(IMG_SIZE)

# target_layer: última camada da layer4 da ResNet18
gradcam = GradCAM(model, model.model.layer4[-1])

for image_path in selected:
    true_class = (
        "Benign" if "benign" in image_path.lower() else "Malignant"
    )

    # Pré-processamento
    pil_image    = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Gera CAM
    cam = gradcam.generate(input_tensor)

    # Predição
    with torch.no_grad():
        outputs       = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        print("\nProbabilidades:")
        print(probabilities.cpu().numpy())

    pred_idx    = probabilities.argmax(dim=1).item()
    pred_class  = CLASS_NAMES[pred_idx].capitalize()
    confidence  = probabilities.max().item()
    correct     = "✅" if pred_class.lower() == true_class.lower() else "❌"

    print(f"\n  Imagem:    {os.path.basename(image_path)}")
    print(f"  Real:      {true_class}")
    print(f"  Predição:  {pred_class}  ({confidence*100:.8f}%)  {correct}") 
    print(f"  CAM — min:{cam.min():.3f}  max:{cam.max():.3f}  "
          f"média:{cam.mean():.3f}")

    # Overlay
    original_resized = pil_image.resize((IMG_SIZE, IMG_SIZE))
    overlay          = overlay_gradcam(original_resized, cam)

    # Figura com 3 painéis
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Grad-CAM — Real: {true_class} | "
        f"{correct} Pred: {pred_class} ({confidence*100:.1f}%)",
        fontsize=12, fontweight="bold",
    )

    axes[0].imshow(original_resized)
    axes[0].set_title("Original"); axes[0].axis("off")

    im = axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Heatmap Grad-CAM"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Sobreposição"); axes[2].axis("off")

    plt.tight_layout()

    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(GRADCAM_DIR, f"gradcam_{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Salvo: {save_path}")