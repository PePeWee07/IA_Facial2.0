# 🔍 Reconocimiento Facial con Comparación de Encodings

Este script se encarga de reconocer un rostro a partir de una imagen de entrada, comparando su encoding facial con los almacenados en la base de datos MongoDB. El proceso se basa en calcular la similitud del coseno entre el encoding de la imagen a reconocer y los encodings de la base de datos.

---

## ⚙️ Componentes Principales

### 1. Umbral de Similitud
- **`SIMILARITY_THRESHOLD = 0.92`**  
  Este valor determina el umbral mínimo que debe superar la similitud del coseno para considerar que dos rostros son una coincidencia. Puedes ajustar este valor según tus pruebas para optimizar la precisión.

### 2. Función `cosine_similarity`
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
