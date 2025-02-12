import cv2
import numpy as np
from pymongo import MongoClient
from util.face_utils import load_and_preprocess_image, detect_and_align_multiple, extract_encoding
import time

# Umbral para considerar una coincidencia
SIMILARITY_THRESHOLD = 0.92

# Bandera para activar o desactivar el modo debug
DEBUG = True

# Procesar cada N-ésimo frame (por ejemplo, cada 3° frame)
SKIP_FRAMES = 10

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.reconocimiento
    personas_collection = db.personas

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        exit()

    if DEBUG:
        frame_count = 0
        start_time = time.time()

    trackers = []
    last_detection = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if DEBUG:
            frame_start = time.time()

        frame_count += 1

        # Procesar solo cada N-ésimo frame
        if frame_count % SKIP_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detect_and_align_multiple(rgb_frame)
            trackers = []
            last_detection = []
            for aligned_face, box in faces:
                try:
                    encoding = extract_encoding(aligned_face)
                except Exception as e:
                    print(f"Error extracting encoding: {e}")
                    continue

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
                    color = (0, 255, 0)  # Verde: reconocido
                else:
                    name_label = "Desconocido"
                    color = (0, 0, 255)  # Rojo: desconocido

                last_detection.append((box, name_label, color))
                
                # Crear tracker para el rostro utilizando CSRT. 
                try:
                    tracker = cv2.TrackerCSRT_create()
                except AttributeError:
                    tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, tuple(box))
                trackers.append((tracker, name_label, color))
        else:
            # En frames intermedios, actualizar los trackers
            new_results = []
            for tracker, name_label, color in trackers:
                ok, new_box = tracker.update(frame)
                if ok:
                    new_box = [int(v) for v in new_box]
                    new_results.append((new_box, name_label, color))
            if new_results:
                last_detection = new_results

        # Dibujar los resultados en el frame
        for box, name_label, color in last_detection:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, name_label, (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if DEBUG:
            frame_end = time.time()
            processing_time = frame_end - frame_start
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"Frame {frame_count}: Tiempo de procesamiento (frame de detección): {processing_time:.4f} s | FPS: {fps:.2f}")

        cv2.imshow("Reconocimiento Facial en Tiempo Real", frame)
        key = cv2.waitKey(1)
        if key == 27:  # Presiona Esc para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
