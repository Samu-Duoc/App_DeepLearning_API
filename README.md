# API de reconocimiento facial

## Descripción

Esta API permite identificar personas a partir de una fotografía mediante un
modelo de reconocimiento facial entrenado como clasificador. La aplicación
móvil envía una imagen a la API y recibe una respuesta con el resultado de la
predicción.

La API utiliza el modelo entrenado y el catálogo de personas definido en
`sujetos.csv` para determinar si una persona puede ser identificada.

## Tecnologías utilizadas

- Python
- FastAPI
- Uvicorn
- PyTorch
- Torchvision
- Pillow
- NumPy
- `facenet_pytorch` para la detección del rostro
- SQLite en el flujo auxiliar de construcción de embeddings

## Requisitos

- Python instalado.
- Dependencias especificadas en `requirements.txt`.
- `modelo_entrenado_v2.pt`, que contiene el modelo utilizado por `api.py`.
- `sujetos.csv`, que relaciona los identificadores de clase con los nombres de
  las personas.

## Instalación inicial

Desde la carpeta del proyecto, instala las dependencias:

```powershell
pip install -r requirements.txt
```

Si se utiliza un entorno virtual, se puede crear y activar antes de instalar
las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Construcción de la base de datos

El proyecto incluye `build_db.py`, que construye `faces.db` a partir del
dataset externo configurado en ese script, del archivo `sujetos.csv` y del
modelo `mejor_modelo.pt`.

```powershell
python build_db.py
```

Este proceso genera tablas de sujetos y embeddings. La base de datos se utiliza
en el flujo auxiliar de embeddings; `api.py` carga directamente
`modelo_entrenado_v2.pt` y `sujetos.csv`, por lo que `faces.db` no es necesaria
para iniciar el endpoint `/predecir/`.

## Iniciar la API

Desde la carpeta que contiene `api.py`, ejecuta:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

La API quedará disponible en el puerto `8000` del computador.

Para consultar la documentación interactiva de FastAPI, abre:

`http://127.0.0.1:8000/docs`

## Modelo y procesamiento de la imagen

El checkpoint actual (`modelo_entrenado_v2.pt`) utiliza la arquitectura
`cnn_bn_dropout`, contiene 37 clases y declara un tamaño de imagen de `128`.
La API procesa la imagen como RGB y aplica las mismas constantes de
normalización utilizadas durante el entrenamiento:

- `MEAN = [0.485, 0.456, 0.406]`
- `STD = [0.229, 0.224, 0.225]`

En el flujo de archivo, la imagen se abre con Pillow, se convierte a RGB,
`MTCNN` detecta el rostro y se recorta el rostro con un margen. Después, el
recorte se redimensiona a **128 × 128 píxeles con 3 canales RGB**, se convierte
a tensor, se normaliza y se clasifica con el modelo.

La clase con mayor probabilidad se considera registrada cuando su confianza es
igual o superior a `0.60` (60 %). Si la confianza es inferior, la API responde
que la persona no está registrada.

## Endpoint de reconocimiento

**Método:** `POST`

**Ruta:** `/predecir/`

**URL de ejemplo:**

`http://192.168.1.175:8000/predecir/`

La dirección corresponde a la IP configurada actualmente en la aplicación
móvil. Si la API se ejecuta en otro computador o red, la URL debe utilizar la
IP local de ese computador.

### Parámetros de entrada

La imagen se envía mediante `multipart/form-data`, utilizando el campo:

`archivo`

Ejemplo conceptual de la solicitud:

```text
POST /predecir/
Content-Type: multipart/form-data
archivo=<archivo de imagen>
```

### Funcionamiento

1. La API recibe la fotografía enviada por la aplicación móvil.
2. Lee el archivo y valida que Pillow pueda abrirlo como imagen.
3. Convierte la imagen a RGB.
4. Detecta y recorta el rostro con MTCNN.
5. Prepara la entrada en formato 128×128×3 y la normaliza.
6. El modelo realiza la clasificación y calcula las probabilidades.
7. La confianza máxima se compara con el umbral del 60 %.
8. Se devuelve una respuesta JSON con el resultado de la predicción.

## Respuesta del endpoint

La respuesta real incluye el resultado de la operación, el nombre o estado de
registro, la confianza y un mensaje descriptivo.

Ejemplo de una identificación exitosa:

```json
{
  "success": true,
  "personName": "Carlos Vasquez",
  "confidence": 87.5,
  "isRegistered": true,
  "message": "Persona identificada correctamente.",
  "rostro_detectado": true,
  "predictedClass": "cv_01",
  "predictedClassName": "Carlos Vasquez",
  "predictedConfidence": 87.5,
  "secondClass": "bg_01",
  "secondConfidence": 4.2,
  "nombre_archivo": "foto.jpg"
}
```

Los valores numéricos dependen de la fotografía. Los campos y tipos del
ejemplo corresponden a la respuesta generada por `api.py`.

Cuando la confianza es inferior al umbral, la respuesta conserva
`success: true`, pero indica que la persona no está registrada:

```json
{
  "success": true,
  "personName": "No registrada",
  "confidence": 42.3,
  "isRegistered": false,
  "message": "La persona no está en la base de datos.",
  "rostro_detectado": true,
  "predictedClass": "cv_01",
  "predictedClassName": "Carlos Vasquez",
  "predictedConfidence": 42.3,
  "secondClass": "bg_01",
  "secondConfidence": 18.1,
  "nombre_archivo": "foto.jpg"
}
```

Si no se detecta un rostro, la API devuelve:

```json
{
  "success": true,
  "personName": "No registrada",
  "confidence": 0,
  "isRegistered": false,
  "message": "No se detectó un rostro en la foto",
  "rostro_detectado": false,
  "nombre_archivo": "foto.jpg"
}
```

### Errores implementados

- `400 Bad Request`: el archivo recibido no es una imagen válida.
- `422 Unprocessable Entity`: error de validación de FastAPI, por ejemplo si
  falta el campo obligatorio `archivo`.

## Endpoint alternativo de matriz

El archivo `api.py` también define `POST /predecir-matriz/`. Este endpoint no es
el que utiliza actualmente la aplicación móvil.

Recibe JSON con una matriz de forma exacta `[128][128][3]`, donde cada píxel
contiene valores RGB enteros entre 0 y 255:

```json
{
  "matrix": [
    [[255, 0, 0], [0, 255, 0]],
    [[0, 0, 255], [255, 255, 255]]
  ]
}
```

El ejemplo anterior es solo ilustrativo de la estructura; para ser aceptado,
`matrix` debe contener realmente 128 filas, 128 columnas y 3 canales por
píxel. Si la forma no es `128x128x3`, responde con estado `422`.

## Integración con la aplicación móvil

La aplicación configura la URL de la API en:

`src/services/api.ts`

La configuración actual es:

```ts
export const API_URL = "http://192.168.1.175:8000";
```

La app agrega `/predecir/`, crea un `FormData` y envía la fotografía en el
campo `archivo`. El celular y el computador deben poder comunicarse a través de
la red. Cuando se utiliza una conexión Wi-Fi local, ambos dispositivos deben
estar conectados a una red que permita esa comunicación.

## Archivos principales

- `api.py`: define los endpoints, carga el modelo, detecta y recorta rostros,
  procesa las imágenes y genera las respuestas.
- `modelos.py`: define el tamaño de entrada, la normalización y las
  arquitecturas disponibles.
- `modelo_entrenado_v2.pt`: contiene el modelo entrenado utilizado actualmente
  por la API.
- `sujetos.csv`: relaciona los identificadores de clase con los nombres
  originales de las personas.
- `requirements.txt`: especifica las dependencias de Python.
- `build_db.py`: construye `faces.db` y sus embeddings para el flujo auxiliar.
- `embedding_utils.py`: contiene las transformaciones y funciones de
  extracción de embeddings usadas por `build_db.py`.
- `calibrar_umbral.py`: calcula un umbral informativo para similitudes de
  embeddings; no modifica el umbral de confianza usado por `api.py`.
- `train.py`: entrena y evalúa las arquitecturas; no se ejecuta al iniciar la
  API.
- `faces.db`: base SQLite generada por `build_db.py`, si se ejecuta ese flujo.

## Flujo general

**Aplicación móvil → captura o selección de fotografía → envío como
`multipart/form-data` → API FastAPI → conversión a RGB y detección de rostro →
redimensionamiento a 128×128×3 → normalización → clasificación → comparación
con el umbral de confianza → respuesta JSON → presentación del resultado en la
aplicación.**

## Consideraciones

Los nombres de archivos, las respuestas, los criterios de identificación y los
pasos de instalación deben mantenerse actualizados de acuerdo con la versión
real del código. En particular, el endpoint utilizado por la aplicación móvil
es `/predecir/` y la entrada final al modelo es una imagen RGB de 128×128×3.
