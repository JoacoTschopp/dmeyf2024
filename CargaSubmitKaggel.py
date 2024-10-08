import os
import json
import kaggle

# Configuración de la competencia de Kaggle
COMPETITION_NAME = "dm-ey-f-2024-primera"

# Directorio donde se encuentran los archivos a subir
UPLOAD_DIR = "/home/tu_usuario/buckets/b1/exp/KA7250"

# Ruta del archivo kaggle.json
KAGGLE_JSON_FILE = "/home/tu_usuario/buckets/b1/kaggle.json"

# Carga del archivo kaggle.json
with open(os.path.expanduser(KAGGLE_JSON_FILE), "r") as f:
    kaggle_json = json.load(f)

# Configuración de la API de Kaggle
kaggle.api.authenticate()

# Carga de los archivos en Kaggle
for file_path in os.listdir(UPLOAD_DIR):
    if file_path.endswith(".csv"):
        file_name = os.path.basename(file_path)
        print(f"Cargando archivo {file_name}...")
        kaggle.api.competition_submit_file(os.path.join(UPLOAD_DIR, file_path), COMPETITION_NAME, message="Experimento1 KA7250 Dataset original sin Drifting", quiet=False)

