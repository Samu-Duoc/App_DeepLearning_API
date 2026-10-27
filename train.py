import copy
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from modelos import ARQUITECTURAS, IMG_SIZE, MEAN, STD, construir_modelo

# En Colab el dataset se descomprime en /content; en el PC se lee de la carpeta local
EN_COLAB = os.path.isdir("/content/drive/MyDrive")
DATASET_DIR = ("/content/dataset_local" if EN_COLAB
               else r"C:\Users\USER\Desktop\DEEP LEARNING\proyecto-integracion\data\images")

EPOCAS = 40
BATCH = 16
LR = 1e-3
PACIENCIA = 10  # early stopping sobre F1 de validación
SEED = 42
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


class DatasetSujetos(ImageFolder):
    """ImageFolder que solo toma carpetas de sujeto con formato ab_01."""
    def find_classes(self, directory):
        clases = sorted(e.name for e in os.scandir(directory)
                        if e.is_dir() and re.fullmatch(r"[a-z]{2}_\d{2}", e.name))
        return clases, {c: i for i, c in enumerate(clases)}


tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Dos vistas del mismo dataset: con augmentation (train) y sin (val/test)
ds_train_full = DatasetSujetos(DATASET_DIR, transform=tf_train)
ds_eval_full = DatasetSujetos(DATASET_DIR, transform=tf_eval)
clases = ds_train_full.classes
n_clases = len(clases)
targets = np.array(ds_train_full.targets)
print(f"Sujetos: {n_clases} | Imágenes: {len(targets)}")

indices = np.arange(len(targets))
idx_tv, idx_test = train_test_split(
    indices, test_size=0.2, stratify=targets, random_state=SEED)
idx_train, idx_val = train_test_split(
    idx_tv, test_size=0.25, stratify=targets[idx_tv], random_state=SEED)
print(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

dl_train = DataLoader(Subset(ds_train_full, idx_train), batch_size=BATCH, shuffle=True)
dl_val = DataLoader(Subset(ds_eval_full, idx_val), batch_size=BATCH)
dl_test = DataLoader(Subset(ds_eval_full, idx_test), batch_size=BATCH)


def evaluar(modelo, loader, criterio):
    modelo.eval()
    perdida, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dispositivo), y.to(dispositivo)
            salida = modelo(x)
            perdida += criterio(salida, y).item() * len(y)
            y_true += y.tolist()
            y_pred += salida.argmax(1).tolist()
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return perdida / len(y_true), acc, f1, y_true, y_pred


def entrenar(nombre):
    modelo = construir_modelo(nombre, n_clases, preentrenado=True).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(
        [p for p in modelo.parameters() if p.requires_grad], lr=LR)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    mejor_f1, mejor_estado, sin_mejora = -1, None, 0

    for epoca in range(EPOCAS):
        modelo.train()
        if nombre == "mobilenet_v2":
            modelo.features.eval()  # extractor congelado también en BatchNorm
        perdida, aciertos, total = 0.0, 0, 0
        for x, y in dl_train:
            x, y = x.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad()
            salida = modelo(x)
            loss = criterio(salida, y)
            loss.backward()
            optimizador.step()
            perdida += loss.item() * len(y)
            aciertos += (salida.argmax(1) == y).sum().item()
            total += len(y)
        v_loss, v_acc, v_f1, _, _ = evaluar(modelo, dl_val, criterio)
        hist["train_loss"].append(perdida / total)
        hist["train_acc"].append(aciertos / total)
        hist["val_loss"].append(v_loss)
        hist["val_acc"].append(v_acc)
        print(f"[{nombre}] época {epoca+1:02d} "
              f"loss {perdida/total:.3f} acc {aciertos/total:.2f} | "
              f"val_loss {v_loss:.3f} val_acc {v_acc:.2f} val_f1 {v_f1:.2f}")

        if v_f1 > mejor_f1:
            mejor_f1, mejor_estado, sin_mejora = v_f1, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                print(f"[{nombre}] early stopping")
                break

    modelo.load_state_dict(mejor_estado)
    return modelo, hist, mejor_f1


def graficar_curvas(nombre, hist):
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].plot(hist["train_loss"], label="train")
    ejes[0].plot(hist["val_loss"], label="validación")
    ejes[0].set_title(f"{nombre} - Pérdida"); ejes[0].set_xlabel("época"); ejes[0].legend()
    ejes[1].plot(hist["train_acc"], label="train")
    ejes[1].plot(hist["val_acc"], label="validación")
    ejes[1].set_title(f"{nombre} - Accuracy"); ejes[1].set_xlabel("época"); ejes[1].legend()
    plt.tight_layout()
    plt.savefig(f"curvas_{nombre}.png", dpi=130)
    plt.close()


resultados, modelos_entrenados = [], {}
criterio = nn.CrossEntropyLoss()
for nombre in ARQUITECTURAS:
    modelo, hist, val_f1 = entrenar(nombre)
    graficar_curvas(nombre, hist)
    _, t_acc, t_f1, y_true, y_pred = evaluar(modelo, dl_test, criterio)
    n_params = sum(p.numel() for p in modelo.parameters())
    resultados.append({
        "modelo": nombre, "parametros": n_params, "epocas": len(hist["val_acc"]),
        "val_f1_macro": round(val_f1, 3), "test_accuracy": round(t_acc, 3),
        "test_f1_macro": round(t_f1, 3)})
    modelos_entrenados[nombre] = (modelo, y_true, y_pred)

tabla = pd.DataFrame(resultados)
tabla.to_csv("comparacion_modelos.csv", index=False)
print("\n", tabla.to_string(index=False))

# La selección se hace con VALIDACIÓN; el test queda solo para reportar
mejor = tabla.sort_values("val_f1_macro", ascending=False).iloc[0]["modelo"]
print(f"\nMejor modelo (por F1 de validación): {mejor}")

modelo, y_true, y_pred = modelos_entrenados[mejor]
reporte = classification_report(
    y_true, y_pred, labels=list(range(n_clases)),
    target_names=clases, zero_division=0)
print(reporte)
with open("reporte_test.txt", "w", encoding="utf-8") as f:
    f.write(f"Modelo: {mejor}\n\n{reporte}")

cm = confusion_matrix(y_true, y_pred, labels=list(range(n_clases)))
plt.figure(figsize=(10, 9))
plt.imshow(cm, cmap="Blues")
plt.title(f"Matriz de confusión (test) - {mejor}")
plt.xlabel("Predicho"); plt.ylabel("Real")
plt.colorbar(); plt.tight_layout()
plt.savefig(f"matriz_confusion_{mejor}.png", dpi=130)
plt.close()

torch.save({"arquitectura": mejor, "clases": clases,
            "state_dict": modelo.state_dict()}, "mejor_modelo.pt")
print("Guardado: mejor_modelo.pt")