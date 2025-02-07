import os
import numpy as np
from pymongo import MongoClient
from face_utils import cargar_y_preprocesar_imagen, detectar_y_alinear, extraer_encoding

# Umbral para considerar una coincidencia (ajústalo según tus pruebas)
SIMILARITY_THRESHOLD = 0.92

def cosine_similarity(a, b):
    """
    Calcula la similitud de coseno entre dos vectores.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main(image_path):
    client = MongoClient('mongodb://localhost:27017/')
    db = client.reconocimiento
    personas_collection = db.personas

    if not os.path.exists(image_path):
        print(f"La imagen '{image_path}' no existe en la raíz del proyecto.")
        return

    try:
        image_np = cargar_y_preprocesar_imagen(image_path)
        aligned_face = detectar_y_alinear(image_np)
        input_encoding = extraer_encoding(aligned_face)
        print("Encoding obtenido de la imagen a reconocer.")
    except Exception as e:
        print(f"Error en el procesamiento de la imagen: {e}")
        return

    documentos = list(personas_collection.find({}))
    if not documentos:
        print("No se encontraron registros en la base de datos.")
        return

    best_similarity = -1
    best_match = None

    for doc in documentos:
        stored_encoding = np.array(doc["encodings"])
        similarity = cosine_similarity(input_encoding, stored_encoding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = doc

    if best_similarity > SIMILARITY_THRESHOLD:
        print(f"Coincidencia encontrada: {best_match['name']} (similitud: {best_similarity:.2f})")
    else:
        print("No se encontró coincidencia.")

if __name__ == "__main__":
    image_path = "assets/img/Henry_Cavill.jpg"
    main(image_path)
