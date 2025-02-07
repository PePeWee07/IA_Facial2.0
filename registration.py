import os
import cv2
from pymongo import MongoClient
from face_utils import cargar_y_preprocesar_imagen, detectar_y_alinear, extraer_encoding

def register_person(image_path, name, face_id):
    """
    Registra una persona en la base de datos:
    - Procesa la imagen (carga, preprocesa, detecta y alinea el rostro)
    - Extrae el encoding facial
    - Guarda la imagen procesada en una ruta definida
    - Inserta en MongoDB un documento con el encoding, la ruta y el nombre
    """
    try:
        image_np = cargar_y_preprocesar_imagen(image_path)
        aligned_face = detectar_y_alinear(image_np)
        encoding = extraer_encoding(aligned_face)
    except Exception as e:
        print(f"Error en el procesamiento de la imagen: {e}")
        return

    # Generar un nombre seguro para la imagen, eliminando espacios
    safe_name = name.replace(" ", "_")
    face_filename = os.path.join("assets/img/faces", f"{safe_name}_{face_id}.png")
    face_filename = face_filename.replace("\\", "/")
    cv2.imwrite(face_filename, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
    print(f"Rostro alineado guardado en: {face_filename}")

    face_data = {
        "face_id": face_id,
        "image_path": face_filename,
        "encodings": encoding.tolist(),
        "name": name
    }

    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client.reconocimiento
        personas_collection = db.personas
        result = personas_collection.insert_one(face_data)
        print(f"Documento insertado en MongoDB con id: {result.inserted_id}")
    except Exception as e:
        print(f"Error al insertar en MongoDB: {e}")

if __name__ == "__main__":
    # image_path = input("Ingresa la ruta de la imagen de registro: ").strip()
    # name = input("Ingresa el nombre de la persona: ").strip()
    face_id = 7
    image_path = "assets/img/spiderman.jpg"
    name = "spiderman"
    register_person(image_path, name, face_id)
