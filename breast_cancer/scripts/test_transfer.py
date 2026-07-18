import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from dataset.config import MODEL_DIR, PLOTS_DIR
from dataset.dataloader import create_dataloaders
from models.transfer import TransferLearningModel
from utils.evaluate import evaluate_model

os.makedirs(PLOTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 40)
print("TESTE DA RESNET18")
print("=" * 40)

_, _, test_loader = create_dataloaders()

model = TransferLearningModel().to(device)
model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "best_resnet18.pth"), map_location=device)
)

results = evaluate_model(model, test_loader, device)
m       = results["metrics"]

print(f"\nAcurácia    : {m['accuracy']:.4f}")
print(f"Precisão    : {m['precision']:.4f}")
print(f"Recall      : {m['recall']:.4f}")
print(f"Specificity : {m['specificity']:.4f}")
print(f"F1-score    : {m['f1']:.4f}")
if m.get("auc_roc"):
    print(f"AUC-ROC     : {m['auc_roc']:.4f}")
print(f"\nTP:{m['tp']}  TN:{m['tn']}  FP:{m['fp']}  FN:{m['fn']}")

disp = ConfusionMatrixDisplay(
    confusion_matrix=m["confusion_matrix"],
    display_labels=["Benign", "Malignant"],
)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão — ResNet18")
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_resnet18.png"),
            dpi=300, bbox_inches="tight")
plt.show()