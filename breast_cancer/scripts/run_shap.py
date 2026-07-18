import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from dataset.config import (
    DATASET_PATH,
    MODEL_DIR,
    SHAP_DIR,
    IMG_SIZE,
    CLASS_NAMES,
    SEED,
)
from preprocessing.preprocessing import get_test_transform
from models.transfer import TransferLearningModel
from explainability.shap import SHAPExplainer

os.makedirs(SHAP_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 50)
print("SHAP — ResNet18")
print("=" * 50)
print("Dispositivo:", device)

# ── Modelo ────────────────────────────────────────────────────
model = TransferLearningModel().to(device)
model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "best_resnet18.pth"),
               map_location=device)
)
model.eval()

transform = get_test_transform(IMG_SIZE)

# ── Background: 50 imagens ALEATÓRIAS e VARIADAS do dataset ──
# IMPORTANTE: o background NÃO pode ser a mesma imagem explicada.
# Ele representa a "distribuição de referência" do SHAP.
# Usar imagens variadas garante explicações significativas.
print("\n📦 Carregando background SHAP (50 imagens aleatórias)...")

full_dataset    = ImageFolder(root=DATASET_PATH, transform=transform)
random.seed(SEED)
bg_indices      = random.sample(range(len(full_dataset)), 50)
bg_loader       = DataLoader(
    Subset(full_dataset, bg_indices),
    batch_size=50, shuffle=False
)
background, _   = next(iter(bg_loader))
background      = background.to(device)

print(f"  Background shape: {background.shape}")

# ── Imagens a explicar: 1 benigna + 1 maligna ─────────────────
image_paths = [
    os.path.join(root, f)
    for root, _, files in os.walk(DATASET_PATH)
    for f in files
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
]

selected = []
for cls in CLASS_NAMES:
    candidates = [p for p in image_paths if cls in p.lower()]
    if candidates:
        selected.append(random.choice(candidates))

# ── Explainer (criado uma vez, reutilizado para todas) ────────
explainer = SHAPExplainer(model, background)

for image_path in selected:
    true_class = (
        "Benign" if "benign" in image_path.lower() else "Malignant"
    )

    pil_image    = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Predição
    with torch.no_grad():
        outputs       = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
    pred_idx   = probabilities.argmax(dim=1).item()
    pred_class = CLASS_NAMES[pred_idx].capitalize()
    confidence = probabilities.max().item()
    correct    = "✅" if pred_class.lower() == true_class.lower() else "❌"

    print(f"\n  Imagem:   {os.path.basename(image_path)}")
    print(f"  Real:     {true_class}")
    print(f"  Predição: {pred_class} ({confidence*100:.8f}%) {correct}")

    # SHAP values
    print("  Calculando SHAP values...")
    values = explainer.generate(input_tensor)  # (H, W, 3)

    # Imagem original para visualização
    original = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))

    # Mapa de importância: magnitude média pelos canais RGB
    importance = np.abs(values).mean(axis=2)
    importance -= importance.min()
    importance /= (importance.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(importance * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.6 * original + 0.4 * heatmap, 0, 255).astype(np.uint8)

    # Figura com 4 painéis
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"SHAP — Real: {true_class} | "
        f"{correct} Pred: {pred_class} ({confidence*100:.1f}%)\n"
        f"Vermelho = contribui para Malignant | Azul = contribui para Benign",
        fontsize=11, fontweight="bold",
    )

    axes[0].imshow(original)
    axes[0].set_title("Original"); axes[0].axis("off")

    # SHAP com direção (vermelho/azul)
    shap_agg  = values.sum(axis=2)            # soma RGB com sinal
    vmax      = np.abs(shap_agg).max()
    im1 = axes[1].imshow(shap_agg, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("SHAP (com direção)"); axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Magnitude
    im2 = axes[2].imshow(importance, cmap="hot")
    axes[2].set_title("Magnitude SHAP"); axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(overlay)
    axes[3].set_title("Sobreposição"); axes[3].axis("off")

    plt.tight_layout()

    filename  = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(SHAP_DIR, f"shap_{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Salvo: {save_path}")