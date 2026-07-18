import torch.nn as nn


class BreastCancerCNN(nn.Module):
    """
    CNN baseline para classificação de câncer de mama (BreaKHis).

    Arquitetura: 4 blocos Conv → BN → ReLU → MaxPool
                 + AdaptiveAvgPool → Flatten → FC → Dropout → FC

    Entrada: (B, 3, 224, 224)
    Saída:   (B, 2) — logits para [benign, malignant]

    Nota: a saída são LOGITS (sem softmax).
    O softmax é aplicado externamente em evaluate.py.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Bloco 1 — 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 224 → 112

            # Bloco 2 — 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 112 → 56

            # Bloco 3 — 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 56 → 28

            # Bloco 4 — 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 28 → 14

            # Redução espacial → (B, 256, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2),                     # 2 classes: benign / malignant
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x                                   # Retorna LOGITS

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)