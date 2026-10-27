"""
API de reconocimiento facial (FastAPI + PyTorch).
  GET  /                  estado del servidor
  POST /predecir/         recibe un archivo de imagen
  POST /predecir-matriz/  recibe matriz 128x128x3 en JSON (usada por la app)
Responde el nombre de la persona o que no está en la base de datos.
"""
import csv
import io
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from modelos import IMG_SIZE, MEAN, STD, construir_modelo

RUTA_BASE = Path(__file__).resolve().parent
RUTA_CSV = RUTA_BASE.parent / "proyecto-integracion" / "sujetos.csv"
RUTA_MODELO = RUTA_BASE / "modelo_entrenado_v2.pt"
UMBRAL_CONFIANZA = 0.60  # bajo este valor se responde "no registrada"

# 1. Cargar el modelo entrenado
checkpoint = torch.load(RUTA_MODELO, map_location="cpu")
claves_requeridas = {"arquitectura", "clases", "state_dict"}
faltantes = claves_requeridas - checkpoint.keys()
if faltantes:
    raise RuntimeError(
        f"El checkpoint {RUTA_MODELO.name} no contiene: {sorted(faltantes)}"
    )

clases = checkpoint["clases"]
arquitectura = checkpoint["arquitectura"]
tamano_modelo = checkpoint.get("img_size", IMG_SIZE)
if tamano_modelo != IMG_SIZE:
    raise RuntimeError(
        f"El modelo declara img_size={tamano_modelo}, pero la API usa {IMG_SIZE}"
    )

recorte_rostro = checkpoint.get("recorte_rostro", True)
margen_rostro = checkpoint.get("margen", 0.25)
modelo = construir_modelo(arquitectura, len(clases), preentrenado=False)
modelo.load_state_dict(checkpoint["state_dict"])
modelo.eval()
detector = (
    MTCNN(keep_all=True, post_process=False, device="cpu")
    if recorte_rostro
    else None
)

# 2. id_carpeta -> nombre_original
nombres = {}
try:
    with open(RUTA_CSV, encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            nombres[fila["id_carpeta"].strip()] = fila["nombre_original"].strip()
except FileNotFoundError:
    print("sujetos.csv no encontrado: se usará el id de carpeta como nombre")

# 3. Mismas transformaciones que en el entrenamiento
normalizar = transforms.Normalize(MEAN, STD)
transformacion = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalizar,
])

app = FastAPI(title="DUOC UC - Reconocimiento Facial")


def recortar_rostro(img_pil: Image.Image):
    if detector is None:
        return img_pil

    cajas, probabilidades = detector.detect(img_pil)
    if cajas is None or probabilidades is None:
        return None

    indice = int(np.argmax(probabilidades))
    izquierda, arriba, derecha, abajo = cajas[indice]
    ancho = derecha - izquierda
    alto = abajo - arriba
    cx = (izquierda + derecha) / 2
    cy = (arriba + abajo) / 2
    lado = max(ancho, alto) * (1 + 2 * margen_rostro)

    return img_pil.crop((
        cx - lado / 2,
        cy - lado / 2,
        cx + lado / 2,
        cy + lado / 2,
    ))


def clasificar(tensor: torch.Tensor) -> dict:
    """Recibe un tensor (3,128,128) normalizado y arma la respuesta."""
    with torch.no_grad():
        probs = torch.softmax(modelo(tensor.unsqueeze(0)), dim=1)[0]
    valores, indices = torch.topk(probs, k=min(2, len(clases)))
    confianza = float(valores[0])
    idx = int(indices[0])
    id_clase = clases[idx]
    segunda_confianza = float(valores[1]) if len(valores) > 1 else 0.0
    segunda_clase = clases[int(indices[1])] if len(indices) > 1 else None
    diagnostico = {
        "predictedClass": id_clase,
        "predictedClassName": nombres.get(id_clase, id_clase),
        "predictedConfidence": round(confianza * 100, 1),
        "secondClass": segunda_clase,
        "secondConfidence": round(segunda_confianza * 100, 1),
    }

    if confianza < UMBRAL_CONFIANZA:
        return {
            "success": True,
            "personName": "No registrada",
            "confidence": round(confianza * 100, 1),
            "isRegistered": False,
            "message": "La persona no está en la base de datos.",
            "rostro_detectado": True,
            **diagnostico,
        }

    return {
        "success": True,
        "personName": nombres.get(id_clase, id_clase),
        "confidence": round(confianza * 100, 1),
        "isRegistered": True,
        "message": "Persona identificada correctamente.",
        "rostro_detectado": True,
        **diagnostico,
    }


@app.get("/")
def estado_servidor():
    return {"estado": "Servidor activo", "sujetos": len(clases),
            "modelo": arquitectura,
            "modelo_archivo": RUTA_MODELO.name,
            "recorte_rostro": recorte_rostro,
            "img_size": tamano_modelo}


@app.post("/predecir/")
async def predecir_archivo(archivo: UploadFile = File(...)):
    datos = await archivo.read()
    try:
        imagen = Image.open(io.BytesIO(datos)).convert("RGB")
    except Exception:
        raise HTTPException(400, "El archivo no es una imagen válida")
    rostro = recortar_rostro(imagen)
    if rostro is None:
        return {
            "success": True,
            "personName": "No registrada",
            "confidence": 0,
            "isRegistered": False,
            "message": "No se detectó un rostro en la foto",
            "rostro_detectado": False,
        }

    resultado = clasificar(transformacion(rostro))
    resultado["nombre_archivo"] = archivo.filename
    return resultado


class MatrizRequest(BaseModel):
    matrix: list[list[list[int]]]  # [alto][ancho][R,G,B], valores 0-255


@app.post("/predecir-matriz/")
def predecir_matriz(req: MatrizRequest):
    arreglo = np.array(req.matrix, dtype=np.uint8)
    if arreglo.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise HTTPException(422, f"Se esperaba 128x128x3 y llegó {arreglo.shape}")
    tensor = torch.from_numpy(arreglo).permute(2, 0, 1).float() / 255.0
    return clasificar(normalizar(tensor))