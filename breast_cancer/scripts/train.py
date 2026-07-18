import os
import time
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.config import EPOCHS, LEARNING_RATE, MODEL_DIR, PLOTS_DIR
from dataset.dataloader import create_dataloaders
from models.cnn import BreastCancerCNN

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("Treinamento da CNN")
print("=" * 50)
print("Dispositivo:", device)

train_loader, val_loader, _ = create_dataloaders()

print(f"Treino: {len(train_loader.dataset)} | "
      f"Val: {len(val_loader.dataset)}")

model     = BreastCancerCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)


def train_one_epoch():
    model.train()
    running_loss, running_correct = 0.0, 0
    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss    += loss.item()
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
    return (running_loss / len(train_loader),
            running_correct / len(train_loader.dataset))


def validate():
    model.eval()
    running_loss, running_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs         = model(images)
            loss            = criterion(outputs, labels)
            running_loss   += loss.item()
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
    return (running_loss / len(val_loader),
            running_correct / len(val_loader.dataset))


best_acc, best_epoch, best_model = 0.0, 0, None
patience_counter = 0
PATIENCE         = 5

train_losses, val_losses, train_accs, val_accs = [], [], [], []
start = time.time()

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss,   val_acc   = validate()

    scheduler.step(val_acc)
    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc);    val_accs.append(val_acc)

    if val_acc > best_acc:
        best_acc     = val_acc
        best_epoch   = epoch + 1
        best_model   = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("\nEarly Stopping ativado!")
            break

    print(f"Época [{epoch+1}/{EPOCHS}] "
          f"| LR: {optimizer.param_groups[0]['lr']:.6f} "
          f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")

if best_model:
    torch.save(best_model, os.path.join(MODEL_DIR, "best_cnn.pth"))

elapsed = (time.time() - start) / 60
print(f"\n✅ CNN finalizada — Melhor época: {best_epoch} | "
      f"Melhor val_acc: {best_acc:.4f} | Tempo: {elapsed:.1f} min")

# Gráficos
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Treinamento CNN — BreaKHis")
for ax, train_vals, val_vals, title, ylabel in zip(
    axes,
    [train_losses, train_accs],
    [val_losses,   val_accs],
    ["Loss", "Acurácia"],
    ["Loss", "Acurácia"],
):
    ax.plot(train_vals, marker="o", label="Treino")
    ax.plot(val_vals,   marker="s", label="Validação")
    ax.set_title(title); ax.set_xlabel("Época")
    ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cnn_training_curve.png"),
            dpi=300, bbox_inches="tight")
plt.show()

pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses,
              "train_acc": train_accs,   "val_acc": val_accs}
             ).to_csv(os.path.join("results", "cnn_history.csv"), index=False)