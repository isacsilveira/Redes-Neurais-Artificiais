import torch
import numpy as np
from utils.metrics import calculate_metrics
from models.ensemble import EnsembleModel


def evaluate_model(model, dataloader, device):
    """
    Avalia um modelo no conjunto de dados fornecido.

    Trata corretamente CNN/ResNet (saída = logits)
    e EnsembleModel (saída = probabilidades).

    Parameters
    ----------
    model      : nn.Module — modelo treinado
    dataloader : DataLoader — conjunto de avaliação
    device     : torch.device

    Returns
    -------
    dict com métricas, y_true, y_pred e y_prob
    """
    model.eval()

    # Detecta se é ensemble (já retorna probabilidades)
    is_ensemble = isinstance(model, EnsembleModel)

    y_true = []
    y_pred = []
    y_prob = []   # Probabilidade da classe malignant (índice 1)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)

            if is_ensemble:
                # Ensemble já retorna probabilidades — NÃO aplicar softmax de novo
                probabilities = outputs
            else:
                # CNN e ResNet retornam logits — aplicar softmax aqui
                probabilities = torch.softmax(outputs, dim=1)

            predictions = probabilities.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())
            # Salva apenas a prob da classe malignant (índice 1) para AUC-ROC
            y_prob.extend(probabilities[:, 1].cpu().numpy())

    metrics = calculate_metrics(y_true, y_pred, y_prob=y_prob)

    return {
        "metrics": metrics,
        "y_true":  y_true,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
    }