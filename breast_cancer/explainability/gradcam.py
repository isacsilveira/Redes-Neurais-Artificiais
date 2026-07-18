import cv2
import numpy as np
import torch


class GradCAM:
    """
    Implementação do Grad-CAM para visualização das regiões
    mais relevantes na decisão do modelo.

    Correção aplicada:
      - generate() agora usa torch.enable_grad() explicitamente.
        model.eval() desativa o Dropout/BN mas NÃO desativa gradientes.
        Sem enable_grad(), o backward() falha em contextos onde
        torch.no_grad() foi ativado externamente.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None

        # Hook: captura ativações no forward
        self.target_layer.register_forward_hook(self._save_activation)

        # Hook: captura gradientes no backward
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor) -> np.ndarray:
        """
        Gera o mapa de calor Grad-CAM para a classe predita.

        Args:
            image: Tensor (1, 3, H, W) normalizado.

        Returns:
            cam: Array (H, W) com valores em [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # enable_grad garante que o backward funciona
        # mesmo que o chamador esteja dentro de torch.no_grad()
        with torch.enable_grad():
            outputs    = self.model(image)
            prediction = outputs.argmax(dim=1).item()
            score      = outputs[:, prediction]
            score.backward()

        if self.gradients is None:
            raise RuntimeError(
                "Gradientes não capturados. "
                "Verifique se target_layer está no grafo computacional."
            )
        if self.activations is None:
            raise RuntimeError("Ativações não capturadas.")

        # Peso de cada canal = média espacial dos gradientes
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # CAM = soma ponderada dos mapas de ativação
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normaliza para [0, 1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # Redimensiona para o tamanho da imagem de entrada
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))

        return cam


def overlay_gradcam(image, cam: np.ndarray,
                    alpha: float = 0.6) -> np.ndarray:
    """
    Sobrepõe o heatmap Grad-CAM na imagem original.

    Args:
        image: PIL Image ou array RGB uint8.
        cam:   Array (H, W) em [0, 1] retornado por generate().
        alpha: Peso da imagem original (0–1). Padrão: 0.6.

    Returns:
        overlay: Array RGB uint8 com heatmap sobreposto.
    """
    image   = np.array(image).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlay