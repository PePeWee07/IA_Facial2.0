import os
import cv2
from pymongo import MongoClient
from util.face_utils import load_and_preprocess_image, detect_and_align, extract_encoding

def register_person(image_path, name, face_id):
    try:
        image_np = load_and_preprocess_image(image_path)
        aligned_face = detect_and_align(image_np)
        encoding = extract_encoding(aligned_face)
    except Exception as e:
        print(f"Error en el procesamiento de la imagen: {e}")
        return

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
    face_id = 0
    image_path = "assets/img/thor.jpg"
    name = "thor"
    register_person(image_path, name, face_id)
