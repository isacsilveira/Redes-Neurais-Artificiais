import numpy as np
import shap
import torch


class SHAPExplainer:
    """
    Explicações SHAP via GradientExplainer para modelos PyTorch.

    Correção em relação à versão anterior:
      O background deve conter imagens VARIADAS do dataset,
      não a mesma imagem que será explicada.
      Usar a própria imagem como background zera os SHAP values
      e distorce completamente a explicação.

    Uso correto:
        background = carregar_batch_aleatorio(dataset, n=50)
        explainer  = SHAPExplainer(model, background)
        values     = explainer.generate(minha_imagem)
    """

    def __init__(self, model, background: torch.Tensor):
        """
        Args:
            model:      Modelo PyTorch em modo eval.
            background: Tensor (N, 3, H, W) com N imagens de referência.
                        Recomendado: 30–100 imagens variadas do treino.
                        NÃO usar a mesma imagem que será explicada.
        """
        self.model = model
        self.model.eval()

        self.explainer = shap.GradientExplainer(
            self.model,
            background,
        )

    def generate(self, image: torch.Tensor) -> np.ndarray:
        """
        Calcula SHAP values para a classe predita.

        Args:
            image: Tensor (1, 3, H, W) normalizado.

        Returns:
            values: Array (H, W, 3) com contribuições por pixel/canal.
                    Positivo → empurra para malignant.
                    Negativo → empurra para benign.
        """
        shap_values = self.explainer.shap_values(image)

        # Versões antigas do SHAP retornam lista [classe0, classe1]
        if isinstance(shap_values, list):
            with torch.no_grad():
                prediction = self.model(image).argmax(dim=1).item()
            values = shap_values[prediction]
        else:
            values = shap_values

        values = np.squeeze(values)

        # SHAP >= 0.52 retorna (3, H, W, N_classes)
        if values.ndim == 4:
            with torch.no_grad():
                prediction = self.model(image).argmax(dim=1).item()
            values = values[:, :, :, prediction]   # → (3, H, W)

        # Garante formato (H, W, 3) para visualização
        if values.ndim == 3 and values.shape[0] == 3:
            values = np.transpose(values, (1, 2, 0))   # → (H, W, 3)

        return values