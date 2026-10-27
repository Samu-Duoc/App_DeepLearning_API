import sqlite3
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
BASE_DATOS = BASE_DIR / "faces.db"


def main():
    with sqlite3.connect(BASE_DATOS) as conexion:
        filas = [
            (sujeto_id, np.frombuffer(vector, dtype=np.float32))
            for sujeto_id, vector in conexion.execute(
                "SELECT sujeto_id, vector FROM embeddings"
            )
        ]

    grupos = {}
    for sujeto_id, vector in filas:
        grupos.setdefault(sujeto_id, []).append(vector)

    internas = []
    externas = []
    sujetos = list(grupos)
    for indice, sujeto_a in enumerate(sujetos):
        for vector_a in grupos[sujeto_a]:
            for sujeto_b in sujetos[indice:]:
                for vector_b in grupos[sujeto_b]:
                    similitud = float(np.dot(vector_a, vector_b))
                    if sujeto_a == sujeto_b:
                        internas.append(similitud)
                    else:
                        externas.append(similitud)

    if not internas or not externas:
        raise RuntimeError("No hay suficientes embeddings para calibrar.")
    umbral = (np.percentile(externas, 99) + np.percentile(internas, 5)) / 2
    print(f"Misma persona: p5={np.percentile(internas, 5):.4f}")
    print(f"Personas distintas: p99={np.percentile(externas, 99):.4f}")
    print(f"Umbral sugerido: {umbral:.4f}")


if __name__ == "__main__":
    main()
