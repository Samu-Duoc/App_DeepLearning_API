import torch.nn as nn
from torchvision import models

IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ARQUITECTURAS = ["cnn_basica", "cnn_bn_dropout", "mobilenet_v2"]


def construir_modelo(nombre, n_clases, preentrenado=False):
    if nombre == "cnn_basica":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, n_clases))

    if nombre == "cnn_bn_dropout":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Flatten(), nn.Linear(128 * 16 * 16, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_clases))

    if nombre == "mobilenet_v2":
        pesos = models.MobileNet_V2_Weights.DEFAULT if preentrenado else None
        modelo = models.mobilenet_v2(weights=pesos)
        for p in modelo.features.parameters():  # se congela el extractor
            p.requires_grad = False
        modelo.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(modelo.last_channel, n_clases))
        return modelo

    raise ValueError(f"Arquitectura desconocida: {nombre}")