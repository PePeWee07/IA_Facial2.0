import cv2
import numpy as np
from pymongo import MongoClient
from util.face_utils import detect_and_align_multiple, extract_encoding
import time

# Umbral para considerar una coincidencia
SIMILARITY_THRESHOLD = 0.92

# Bandera para activar o desactivar el modo debug
DEBUG = True

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.reconocimiento
    personas_collection = db.personas

    # Inicializar la captura de video (cámara por defecto 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        exit()

    # Inicialización de variables para debug
    if DEBUG:
        frame_count = 0
        start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Si debug está activo, medir el tiempo de procesamiento para este frame
        if DEBUG:
            frame_start = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_and_align_multiple(rgb_frame)
        
        for aligned_face, box in faces:
            try:
                encoding = extract_encoding(aligned_face)
            except Exception as e:
                print(f"Error extracting encoding: {e}")
                continue

            # Comparar el encoding con los registros en la base de datos
            documentos = list(personas_collection.find({}))
            best_similarity = -1
            best_match = None
            for doc in documentos:
                stored_encoding = np.array(doc["encodings"])
                similarity = cosine_similarity(encoding, stored_encoding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = doc

            if best_similarity > SIMILARITY_THRESHOLD:
                name_label = best_match["name"]
                color = (0, 255, 0)  # Verde para rostros reconocidos
            else:
                name_label = "Desconocido"
                color = (0, 0, 255)  # Rojo para rostros desconocidos

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, name_label, (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Si debug está activo, medir y mostrar el rendimiento
        if DEBUG:
            frame_end = time.time()
            processing_time = frame_end - frame_start
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"Frame {frame_count}: Tiempo de procesamiento: {processing_time:.4f} s | FPS: {fps:.2f}")

        cv2.imshow("Reconocimiento Facial en Tiempo Real", frame)
        key = cv2.waitKey(1)
        if key == 27:  # Presiona Esc para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
