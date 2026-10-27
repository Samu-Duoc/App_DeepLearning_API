import io
import sqlite3
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from embedding_utils import TRANSFORMACION, cargar_modelo

BASE_DIR = Path(__file__).resolve().parent
MODELO, CHECKPOINT = cargar_modelo(BASE_DIR / "mejor_modelo.pt")
BASE_DATOS = BASE_DIR / "faces.db"
UMBRAL_SIMILITUD = 0.75
UMBRAL_CONFIANZA = 0.60

app = FastAPI(title="DUOC UC - Reconocimiento facial")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def cargar_embeddings():
    if not BASE_DATOS.exists():
        raise RuntimeError(
            f"No existe {BASE_DATOS}. Ejecuta primero: python build_db.py"
        )
    with sqlite3.connect(BASE_DATOS) as conexion:
        sujetos = dict(conexion.execute(
            "SELECT id, nombre FROM sujetos"
        ).fetchall())
        embeddings = [
            (sujeto_id, np.frombuffer(vector, dtype=np.float32))
            for sujeto_id, vector in conexion.execute(
                "SELECT sujeto_id, vector FROM embeddings"
            )
        ]
    if not embeddings:
        raise RuntimeError("faces.db no contiene embeddings.")
    return sujetos, embeddings


SUJETOS, EMBEDDINGS = cargar_embeddings()


@app.get("/")
def estado():
    return {
        "estado": "Servidor activo",
        "sujetos": len(SUJETOS),
        "embeddings": len(EMBEDDINGS),
        "modelo": CHECKPOINT["arquitectura"],
    }


@app.post("/reconocer")
async def reconocer(archivo: UploadFile = File(...)):
    datos = await archivo.read()
    try:
        imagen = Image.open(io.BytesIO(datos)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, "El archivo recibido no es una imagen válida") from exc

    tensor = TRANSFORMACION(imagen).unsqueeze(0)
    with torch.no_grad():
        probabilidades = torch.softmax(MODELO(tensor), dim=1)[0]
    confianza, indice = probabilidades.max(0)
    confianza = float(confianza)
    mejor_id = CHECKPOINT["clases"][int(indice)]
    registrado = confianza >= UMBRAL_CONFIANZA and mejor_id in SUJETOS
    return {
        "registrado": registrado,
        "nombre": SUJETOS[mejor_id] if registrado else None,
        "id": mejor_id if registrado else None,
        "confianza": round(confianza * 100, 1),
    }
