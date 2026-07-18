from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import numpy as np


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calcula métricas de classificação binária.

    Parameters
    ----------
    y_true : list | array
        Classes reais (0 = benign, 1 = malignant).
    y_pred : list | array
        Classes preditas pelo modelo.
    y_prob : list | array, opcional
        Probabilidades da classe positiva (malignant).
        Se fornecido, calcula AUC-ROC também.

    Returns
    -------
    dict com accuracy, precision, recall, specificity, f1,
         confusion_matrix e (se y_prob) auc_roc.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred, zero_division=0),
        "recall":           recall_score(y_true, y_pred, zero_division=0),
        "specificity":      specificity,
        "f1":               f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }

    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = None

    return metrics