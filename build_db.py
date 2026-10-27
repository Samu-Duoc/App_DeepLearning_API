import csv
import sqlite3
from pathlib import Path

import numpy as np

from embedding_utils import embedding_desde_imagen, cargar_modelo

BASE_DIR = Path(__file__).resolve().parent
MODELO = BASE_DIR / "mejor_modelo.pt"
BASE_DATOS = BASE_DIR / "faces.db"
CSV_SUJETOS = BASE_DIR.parent / "proyecto-integracion" / "sujetos.csv"
IMAGES_DIR = BASE_DIR.parent / "proyecto-integracion" / "data" / "images"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}


def cargar_sujetos():
    with CSV_SUJETOS.open(encoding="utf-8-sig", newline="") as archivo:
        return {
            fila["id_carpeta"].strip(): fila
            for fila in csv.DictReader(archivo)
            if (IMAGES_DIR / fila["id_carpeta"].strip()).is_dir()
        }


def main():
    sujetos = cargar_sujetos()
    modelo, checkpoint = cargar_modelo(MODELO)
    clases = set(checkpoint["clases"])

    with sqlite3.connect(BASE_DATOS) as conexion:
        conexion.executescript("""
            DROP TABLE IF EXISTS embeddings;
            DROP TABLE IF EXISTS sujetos;
            CREATE TABLE sujetos (
                id TEXT PRIMARY KEY,
                nombre TEXT NOT NULL,
                nombre_normalizado TEXT NOT NULL
            );
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sujeto_id TEXT NOT NULL REFERENCES sujetos(id),
                vector BLOB NOT NULL
            );
        """)

        total = 0
        for sujeto_id, fila in sujetos.items():
            if sujeto_id not in clases:
                continue
            conexion.execute(
                "INSERT INTO sujetos (id, nombre, nombre_normalizado) VALUES (?, ?, ?)",
                (sujeto_id, fila["nombre_original"].strip(),
                 fila["nombre_normalizado"].strip()),
            )
            imagenes = sorted(
                imagen for imagen in (IMAGES_DIR / sujeto_id).iterdir()
                if imagen.suffix.lower() in EXTENSIONES
            )
            for imagen in imagenes:
                vector = embedding_desde_imagen(modelo, imagen)
                conexion.execute(
                    "INSERT INTO embeddings (sujeto_id, vector) VALUES (?, ?)",
                    (sujeto_id, vector.tobytes()),
                )
                total += 1
        conexion.commit()

    print(f"Base creada: {BASE_DATOS}")
    print(f"Sujetos incluidos: {len(sujetos)}")
    print(f"Embeddings guardados: {total}")


if __name__ == "__main__":
    main()
