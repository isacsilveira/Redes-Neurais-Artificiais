import os
import torch
import torch.nn as nn

from dataset.config import MODEL_DIR
from models.cnn import BreastCancerCNN
from models.transfer import TransferLearningModel


class EnsembleModel(nn.Module):
    """
    Ensemble Soft Voting ponderado: CNN (40%) + ResNet18 (60%).

    IMPORTANTE — contrato de saída:
      - CNN e ResNet18 retornam LOGITS
      - O EnsembleModel converte para probabilidades internamente
        e retorna PROBABILIDADES (após softmax)
      - Por isso evaluate.py NÃO deve aplicar softmax novamente
        no ensemble — apenas em CNN e ResNet18 individuais

    Correção aplicada:
      Versão anterior: (0.1 * cnn + 0.9 * resnet) / 2
        → pesos não somavam 1.0; divisão por 2 distorcia as probs
      Versão correta:   0.4 * cnn + 0.6 * resnet
        → média ponderada correta; resultado soma 1.0
    """

    def __init__(self, device):
        super().__init__()

        self.device        = device
        self.weight_cnn    = 0.4
        self.weight_resnet = 0.6

        self.cnn    = BreastCancerCNN().to(device)
        self.resnet = TransferLearningModel().to(device)

        self._load_models()

    def _load_models(self):
        cnn_path    = os.path.join(MODEL_DIR, "best_cnn.pth")
        resnet_path = os.path.join(MODEL_DIR, "best_resnet18.pth")

        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"Modelo CNN não encontrado: {cnn_path}")
        if not os.path.exists(resnet_path):
            raise FileNotFoundError(f"Modelo ResNet não encontrado: {resnet_path}")

        self.cnn.load_state_dict(
            torch.load(cnn_path, map_location=self.device)
        )
        self.resnet.load_state_dict(
            torch.load(resnet_path, map_location=self.device)
        )

        self.cnn.eval()
        self.resnet.eval()

        print(f"✅ Ensemble carregado — "
              f"CNN: {self.weight_cnn*100:.0f}% | "
              f"ResNet18: {self.weight_resnet*100:.0f}%")

    def forward(self, x):
        """
        Retorna PROBABILIDADES (não logits).
        Softmax aplicado aqui — evaluate.py não deve aplicar de novo.
        """
        with torch.no_grad():
            cnn_prob    = torch.softmax(self.cnn(x),    dim=1)
            resnet_prob = torch.softmax(self.resnet(x), dim=1)

            # Média ponderada — resultado soma 1.0 por item do batch
            ensemble_prob = (
                self.weight_cnn    * cnn_prob +
                self.weight_resnet * resnet_prob
            )

        return ensemble_prob