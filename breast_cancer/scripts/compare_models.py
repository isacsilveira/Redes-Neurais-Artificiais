import os
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataset.config import MODEL_DIR, RESULTS_DIR, PLOTS_DIR
from dataset.dataloader import create_dataloaders
from models.cnn import BreastCancerCNN
from models.transfer import TransferLearningModel
from models.ensemble import EnsembleModel
from utils.evaluate import evaluate_model

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("COMPARAÇÃO DOS MODELOS")
print("=" * 50)
print("Dispositivo:", device)

_, _, test_loader = create_dataloaders()

# ── Carrega e avalia cada modelo ──────────────────────────────
def load_and_evaluate(model, weights_path, name):
    """Carrega pesos, avalia e retorna métricas."""
    if not os.path.exists(weights_path):
        print(f"⚠️  Modelo não encontrado: {weights_path} — pulando.")
        return None
    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    print(f"\n  Avaliando {name}...")
    results = evaluate_model(model, test_loader, device)
    m       = results["metrics"]
    print(f"  Acc:{m['accuracy']:.4f}  F1:{m['f1']:.4f}  "
          f"AUC:{m.get('auc_roc', 0.0) or 0.0:.4f}")
    return m


cnn_metrics = load_and_evaluate(
    BreastCancerCNN().to(device),
    os.path.join(MODEL_DIR, "best_cnn.pth"),
    "CNN",
)

resnet_metrics = load_and_evaluate(
    TransferLearningModel().to(device),
    os.path.join(MODEL_DIR, "best_resnet18.pth"),
    "ResNet18",
)

# Ensemble carrega os modelos internamente
print("\n  Avaliando Ensemble...")
ensemble_model   = EnsembleModel(device)
ensemble_results = evaluate_model(ensemble_model, test_loader, device)
ensemble_metrics = ensemble_results["metrics"]
print(f"  Acc:{ensemble_metrics['accuracy']:.4f}  "
      f"F1:{ensemble_metrics['f1']:.4f}  "
      f"AUC:{ensemble_metrics.get('auc_roc', 0.0) or 0.0:.4f}")

# ── Monta tabela ──────────────────────────────────────────────
rows = []
for name, m in [("CNN", cnn_metrics),
                ("ResNet18", resnet_metrics),
                ("Ensemble", ensemble_metrics)]:
    if m is None:
        continue
    rows.append({
        "Modelo":      name,
        "Accuracy":    round(m["accuracy"],    4),
        "Precision":   round(m["precision"],   4),
        "Recall":      round(m["recall"],      4),
        "Specificity": round(m["specificity"], 4),
        "F1-score":    round(m["f1"],          4),
        "AUC-ROC":     round(m.get("auc_roc") or 0.0, 4),
    })

df = pd.DataFrame(rows)

print("\n" + "=" * 60)
print(df.to_string(index=False))
print("=" * 60)

csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ comparison.csv salvo: {csv_path}")

# ── Gráfico comparativo ───────────────────────────────────────
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]
x      = range(len(df))
colors = ["#2196F3", "#4CAF50", "#FF5722"]

fig, axes = plt.subplots(1, len(metrics_to_plot),
                          figsize=(4 * len(metrics_to_plot), 5))
fig.suptitle("Comparação dos Modelos — BreaKHis",
             fontsize=13, fontweight="bold")

for ax, metric in zip(axes, metrics_to_plot):
    values = df[metric].tolist()
    bars   = ax.bar(x, values, color=colors[:len(df)],
                    alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.set_title(metric, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Modelo"].tolist(), fontsize=9)
    ax.set_ylim(max(0, min(values) - 0.05), 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center",
                fontsize=8, fontweight="bold")

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "comparison_models.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"✅ Gráfico salvo: {plot_path}")