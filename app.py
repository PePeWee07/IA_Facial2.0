import cv2
import os
import json
import torch
from facenet_pytorch import MTCNN
import face_recognition
from PIL import Image
import numpy as np
from pymongo import MongoClient

# --- CONEXIÓN A MONGODB ---
client = MongoClient('mongodb://localhost:27017/')
db = client.reconocimiento
personas_collection = db.personas

# --- CREACIÓN DE CARPETAS ---
faces_dir = "assets/img/faces"
for directory in [faces_dir]:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Error al crear la carpeta {directory}: {str(e)}")
        exit()

# --- CONFIGURACIÓN DE DISPOSITIVO (GPU o CPU) ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# --- CONFIGURACIÓN DE MTCNN ---
mtcnn = MTCNN(
    select_largest=True,         # Seleccionar el rostro más grande
    min_face_size=100,           # Rostros de al menos 100 píxeles
    thresholds=[0.8, 0.8, 0.98], # Umbrales para la detección
    post_process=False,          # No se aplicará post-procesamiento
    image_size=224,              # Se utilizará para la detección
    margin=10,                   # Margen alrededor del rostro
    device=device                # Seleccionar GPU o CPU
)

# --- FUNCIÓN PARA ALINEAR EL ROSTRO ---
def align_face(image, landmarks):
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
    center = tuple(map(int, np.mean([left_eye, right_eye], axis=0)))
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    return aligned_image

# --- CARGA Y PREPROCESAMIENTO DE LA IMAGEN ---
imagesPath = "assets/img/person2.jpg"

try:
    image = Image.open(imagesPath)
except Exception as e:
    print(f"Error al cargar la imagen: {str(e)}")
    exit()

image = image.convert('RGB')
orig_width, orig_height = image.size
max_dim = 1040  # Límite para el lado más largo
if max(orig_width, orig_height) > max_dim and min(orig_width, orig_height) > 400:
    scale_factor = max_dim / max(orig_width, orig_height)
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    print(f"Imagen redimensionada a {new_width}x{new_height} para optimizar el procesamiento.")

# Convertir la imagen a un array de NumPy
image_np = np.array(image)

# (Opcional) Convertir a tensor y enviarlo a GPU si es necesario.
image_tensor = torch.from_numpy(image_np).to(device)

# --- DETECCIÓN DE ROSTRO Y LANDMARKS ---
boxes, probs, landmarks = mtcnn.detect(image_np, landmarks=True)

if boxes is None or landmarks is None or len(boxes) == 0:
    print("No se detectó ningún rostro en la imagen. Verifica iluminación y resolución.")
    exit()

if len(boxes) > 1:
    print("Se detectaron múltiples rostros. Asegúrate de enviar una imagen tipo carnet con un solo rostro.")
    exit()

# --- PROCESAMIENTO DE LA ÚNICA DETECCIÓN ---
# Extraer la caja delimitadora y los landmarks
box = [int(coord) for coord in boxes[0]]
landmark = landmarks[0]

# Extraer el rostro usando la caja delimitadora
face = image_np[box[1]:box[3], box[0]:box[2], :]

# Ajustar las coordenadas de los landmarks al recorte del rostro
adjusted_landmarks = landmark - np.array([box[0], box[1]])

# Alinear el rostro usando los landmarks ajustados
aligned_face = align_face(face, adjusted_landmarks)

# --- REDIMENSIONAMIENTO DEL ROSTRO ---
face_height, face_width = aligned_face.shape[:2]
if min(face_height, face_width) > 224:
    resize_factor = 224 / min(face_height, face_width)
    new_width = int(face_width * resize_factor)
    new_height = int(face_height * resize_factor)
    aligned_face = cv2.resize(aligned_face, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
else:
    print("El rostro ya tiene dimensiones menores o iguales a 224; no se redimensionará.")

# Guardar el rostro alineado, redimensionado y recortado
face_filename = os.path.join(faces_dir, "face_1.png")
face_filename = face_filename.replace("\\", "/")
cv2.imwrite(face_filename, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
print(f"Rostro alineado guardado en: {face_filename}")

# --- CODIFICACIÓN FACIAL ---
face_encodings = face_recognition.face_encodings(aligned_face)
if not face_encodings:
    print("No se pudo generar la codificación facial. Verifica la calidad o el alineamiento del rostro.")
    exit()

face_data = {
    "face_id": 1,
    "image_path": face_filename,
    "encodings": face_encodings[0].tolist(),
    "name": "John Doe"
}

# --- GUARDADO DE LOS DATOS EN MONGODB ---
result = personas_collection.insert_one(face_data)
print(f"Documento insertado en MongoDB con id: {result.inserted_id}")

