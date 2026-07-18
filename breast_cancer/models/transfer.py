import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TransferLearningModel(nn.Module):
    """
    ResNet18 pré-treinada no ImageNet com fine-tuning em layer3 + layer4.

    Mudanças em relação à versão anterior:
      - Descongela layer3 + layer4 (antes só layer4)
        → mais capacidade de adaptação para histologia
      - Classificador com camada intermediária (FC→ReLU→Dropout→FC)
        → mais expressivo que Dropout→FC direto

    Saída: (B, 2) — LOGITS para [benign, malignant]
    """

    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Congela tudo primeiro
        for param in self.model.parameters():
            param.requires_grad = False

        # Descongela layer3 e layer4 para fine-tuning
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Substitui o classificador original
        in_features   = self.model.fc.in_features   # 512 na ResNet18
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2),                       # Logits: benign / malignant
        )

        # Garante que o novo fc é treinável
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self._initialize_weights()

    def forward(self, x):
        return self.model(x)                         # Retorna LOGITS

    def _initialize_weights(self):
        for m in self.model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)