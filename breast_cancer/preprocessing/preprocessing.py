from torchvision import transforms
from dataset.config import IMG_SIZE


def get_train_transform(img_size: int = IMG_SIZE):
    """
    Transformações para o conjunto de treino.
    Inclui Data Augmentation para aumentar variabilidade.

    Augmentations escolhidas para histologia:
      - Flip horizontal e vertical: válidos (sem orientação canônica)
      - Rotação ±15°: estruturas celulares aparecem em qualquer ângulo
      - ColorJitter leve: simula variação de coloração H&E entre lâminas
      - Sem zoom excessivo: preserva escala das estruturas celulares
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),          # Válido para histologia
        transforms.RandomRotation(15),
        transforms.ColorJitter(                   # Variação de coloração H&E
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],           # Médias do ImageNet
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_test_transform(img_size: int = IMG_SIZE):
    """
    Transformações para validação e teste.
    Apenas resize e normalização — SEM augmentation.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])