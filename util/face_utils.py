import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import face_recognition
import torch

# Configuración global del dispositivo y del detector MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Configuración del detector MTCNN
mtcnn = MTCNN(
    select_largest=True,         # Seleccionar el rostro más grande
    min_face_size=100,           # Rostros de al menos 100 píxeles
    thresholds=[0.8, 0.8, 0.98], # Umbrales para la detección
    post_process=False,          # No se aplicará post-procesamiento
    image_size=224,              # Tamaño de la imagen para la detección
    margin=10,                   # Margen alrededor del rostro
    device=device                # Seleccionar GPU o CPU
)

# Carga y preprocesamiento de la imagen
def load_and_preprocess_image(image_path, max_dim=1040):
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise Exception(f"Error al cargar la imagen: {e}")
    image = image.convert('RGB')
    orig_width, orig_height = image.size
    if max(orig_width, orig_height) > max_dim and min(orig_width, orig_height) > 400:
        scale_factor = max_dim / max(orig_width, orig_height)
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return np.array(image)


# Alineación del rostro
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


# Detección y alineación del rostro
def detect_and_align(image_np):
    try:
        boxes, probs, landmarks = mtcnn.detect(image_np, landmarks=True)
        if boxes is None or landmarks is None or len(boxes) == 0:
            raise Exception("No se detectó ningún rostro en la imagen.")
        if len(boxes) > 1:
            raise Exception("Se detectaron múltiples rostros. Se espera uno solo.")
        box = [int(coord) for coord in boxes[0]]
        landmark = landmarks[0]
        face = image_np[box[1]:box[3], box[0]:box[2], :]
        adjusted_landmarks = landmark - np.array([box[0], box[1]])
        aligned_face = align_face(face, adjusted_landmarks)
        face_height, face_width = aligned_face.shape[:2]
        if min(face_height, face_width) > 224:
            resize_factor = 224 / min(face_height, face_width)
            new_width = int(face_width * resize_factor)
            new_height = int(face_height * resize_factor)
            aligned_face = cv2.resize(aligned_face, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return aligned_face
    except Exception as e:
        raise Exception(f"Error al detectar y alinear el rostro: {e}")


# Extracción del encoding facial
def extract_encoding(aligned_face):
    h, w, _ = aligned_face.shape
    face_locations = [(0, w, h, 0)]
    face_encodings = face_recognition.face_encodings(aligned_face, known_face_locations=face_locations)
    if not face_encodings:
        raise Exception("No se pudo generar la codificación facial.")
    return face_encodings[0]

