from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modelos import IMG_SIZE, MEAN, STD, construir_modelo

TRANSFORMACION = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def cargar_modelo(ruta_modelo: Path):
    checkpoint = torch.load(ruta_modelo, map_location="cpu")
    modelo = construir_modelo(
        checkpoint["arquitectura"],
        len(checkpoint["clases"]),
        preentrenado=False,
    )
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    return modelo, checkpoint


def embedding_desde_tensor(modelo, tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        if hasattr(modelo, "features") and hasattr(modelo, "avgpool"):
            salida = modelo.features(tensor.unsqueeze(0))
            salida = modelo.avgpool(salida).flatten(1)
        else:
            salida = tensor.unsqueeze(0)
            for capa in list(modelo.children())[:-1]:
                salida = capa(salida)
            salida = salida.flatten(1)

    vector = salida[0].numpy().astype(np.float32)
    norma = np.linalg.norm(vector)
    if norma == 0:
        raise ValueError("El modelo produjo un embedding vacío.")
    return vector / norma


def embedding_desde_imagen(modelo, ruta_imagen: Path) -> np.ndarray:
    try:
        imagen = Image.open(ruta_imagen).convert("RGB")
    except Exception as exc:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}") from exc
    return embedding_desde_tensor(modelo, TRANSFORMACION(imagen))
