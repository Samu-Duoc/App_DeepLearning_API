# API de reconocimiento

La API usa `mejor_modelo.pt` como clasificador de personas y `faces.db` como
catálogo SQLite de las personas que tienen imágenes disponibles. El CSV fuente
es:

```text
..\proyecto-integracion\sujetos.csv
```

Se incluyen únicamente los sujetos que tienen una carpeta en:

```text
..\proyecto-integracion\data\images
```

## Instalación

Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Construir la base

```powershell
python build_db.py
```

Esto crea `faces.db` con los 30 sujetos disponibles y sus 300 imágenes.

## Calibración informativa

```powershell
python calibrar_umbral.py
```

El modelo actual fue entrenado como clasificador; por eso la API usa su
confianza de clasificación para decidir el registro. La extracción de
características se conserva en `build_db.py` para dejar disponible la base de
embeddings, pero no se usa como criterio principal porque sus similitudes se
solapan entre sujetos.

## Arrancar la API

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoint móvil (el que consume la app):

```text
POST http://IP_DE_LA_PC:8000/predecir/
```

El campo multipart de la imagen debe llamarse `archivo`.

Respuesta del endpoint `/predecir/`:

```json
{
  "success": true,
  "personName": "Carlos Vasquez",
  "confidence": 87.5,
  "isRegistered": true,
  "message": "Persona identificada correctamente.",
  "rostro_detectado": true,
  "nombre_archivo": "foto.jpg"
}
```

## Expo Go

En `app_DL/src/services/api.ts`, configura la IP local de la PC:

```ts
export const API_URL = "http://192.168.1.175:8000";
```

El celular y la PC deben estar en la misma red Wi-Fi. Luego, desde
`app_DL`, inicia Expo:

```powershell
npx expo start -c
```

La app envía la imagen como `multipart/form-data` a `/predecir/`.
