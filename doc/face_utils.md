## 🌍Configuración global del dispositivo y del detector MTCNN

Este bloque de código se encarga de determinar el dispositivo de cómputo (GPU o CPU).

### Código
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

## ⚙️ Configuración de MTCNN en `facenet-pytorch`

```python
mtcnn = MTCNN(
    select_largest=True,         
    min_face_size=100,           
    thresholds=[0.8, 0.8, 0.98], 
    post_process=False,          
    image_size=224,              
    margin=10,                   
    device=device                
)
```

Para optimizar y mejorar la precisión de tu código al utilizar la clase `MTCNN` del paquete `facenet-pytorch`, es importante comprender las configuraciones disponibles y cómo ajustarlas según tus necesidades. A continuación, se detallan los parámetros clave que puedes considerar:

### 🔧 **Parámetros principales**

- 🔝 **`select_largest`**: Si es `True`, en caso de que se detecten múltiples rostros, se devolverá el más grande. Si es `False`, se devolverá el rostro con la mayor probabilidad de detección. Esto es útil si esperas que haya múltiples rostros en la imagen y deseas centrarte en uno específico.

- 📐 **`min_face_size`**: Establece el tamaño mínimo de rostro que el detector buscará. El valor predeterminado es `20`. Si los rostros en las imágenes son más grandes, aumentar este valor puede reducir falsos positivos y mejorar la eficiencia.

- 🎚️ **`thresholds`**: Son los umbrales de detección para las tres etapas de la red `MTCNN`. Los valores predeterminados son `[0.6, 0.7, 0.7]`. Ajustar estos umbrales puede influir en la sensibilidad y precisión de la detección. Por ejemplo, aumentar los valores puede reducir falsos positivos, pero también podría omitir rostros menos evidentes.

- 🛠️ **`post_process`**: Indica si se debe postprocesar los tensores de imágenes antes de devolverlos. Por defecto, es `True`. Si estás realizando un procesamiento personalizado después de la detección, es posible que desees desactivar esta opción.

    ```text
    En nuestro caso no necesitamos normalización porque la detección de rostros se usa en un flujo de OpenCV + face_recognition. Si estuviéramos pasando los rostros a un modelo preentrenado en PyTorch, podríamos considerar dpost_process=True
    ```

- 📏 **`image_size`**: Define el tamaño de las imágenes de salida en píxeles. Por defecto, es `160`. Si las imágenes de entrada son de alta resolución, ajustar este parámetro puede ayudar a mantener la calidad y detalle necesarios para una detección precisa.

- 🖼️ **`margin`**: Añade un margen al cuadro delimitador en términos de píxeles en la imagen final. Esto es útil para asegurarse de que se capturen áreas adicionales alrededor del rostro, lo que puede ser beneficioso en procesos posteriores como el reconocimiento facial.

- 💻 **`device`**: Especifica el dispositivo en el que se ejecutarán las pasadas de la red neuronal. Los tensores de imágenes y los modelos se copian a este dispositivo antes de ejecutar las pasadas hacia adelante. Por defecto, es `None`, lo que significa que se utilizará la `CPU`, pero si tienes una `GPU` disponible, puedes especificarla para mejorar el rendimiento.

Para una comprensión más profunda y ejemplos prácticos, puedes consultar la guía de `MTCNN` en `facenet-pytorch` y la documentación del [repositorio oficial](https://github.com/timesler/facenet-pytorch).


## 🖼️ Carga y preprocesamiento de la imagen

### 🔍 Descripción de la Función `load_and_preprocess_image`

    función auxiliar que se encarga exclusivamente de alinear el rostro de una imagen recortada.

- Abrir y cargar una imagen desde una ruta específica.

- Convertir la imagen a formato RGB Esto garantiza que la imagen tenga tres canales de color (rojo, verde y azul), independientemente del formato original (por ejemplo, puede venir en modo "L" para escala de grises o "RGBA" con un canal alfa).

- Verifica si es necesario redimensionar la imagen para que no sea demasiado grande (manteniendo la proporción y usando un filtro de alta calidad).

- Retornar la imagen como un array de NumPy.

### 📚 Explicación del Redimensionamiento

### 👉 Condición de Redimensionamiento en el Código
```python
if max(orig_width, orig_height) > max_dim and min(orig_width, orig_height) > 400:
```
Esta condición significa que la imagen **solo se redimensionará si:**
1. **El lado más grande supera 1040px** (`max_dim = 1040`).
2. **El lado más pequeño supera 400px** (`min(orig_width, orig_height) > 400`).

Esto permite optimizar el procesamiento de imágenes sin afectar la calidad de aquellas que ya tienen un tamaño adecuado.

### 📏 Ejemplo de Funcionamiento en Diferentes Casos
| **Tamaño Original** | **Se Redimensiona?** | **Razón** |
|------------------|---------------|--------|
| **1200x800**    | ✅ Sí         | Sí	El lado mayor (1200px) es mayor a 1040px y el lado menor (800px) es mayor a 400px. |
| **1000x650**      | ❌ No         | No	El lado mayor (1000px) es menor que 1040px. |
| **1040x500**      | ❌ No         | No	El lado mayor es igual a 1040px (no lo supera). |
| **1040x300**      | ❌ No         | No	Aunque el lado mayor es 1040px, el lado menor (300px) es inferior a 400px. |
| **600x400**      | ❌ No         | No	Ninguno de los lados supera 1040px, por lo que se mantiene igual. |
| **540x304**      | ❌ No         | No	La imagen es pequeña; mantenerla evita pérdida de calidad. |

### 👉 ¿Por qué no redimensionamos si el lado menor es menor a 400px?
Si una imagen es **demasiado pequeña** (por ejemplo, `1040x300`), es mejor **no cambiar su tamaño**, porque:
✅ **Evita pixelación** → Si ampliamos una imagen con `300px` de altura, perderá calidad.
✅ **Preserva detalles del rostro** → Si la imagen es pequeña, el detector MTCNN ya trabaja con resolución baja, y al modificarla podemos afectar la precisión.

---
### ⚠️ Manejo de Errores
- La función lanza excepciones cuando **Error al cargar la imagen**, facilitando la depuración.
---


## 🤖 Detección y Alineación del Rostro

### 🔍 Descripción de la Función `detect_and_align(image_np)`
    Se encarga de detectar el rostro en una imagen, extraerlo, alinearlo y redimensionarlo para optimizar su procesamiento en tareas posteriores (por ejemplo, la extracción de encodings faciales). A continuación se explica paso a paso su funcionamiento.

### ⚙️ Detalle del Proceso

### 🔍 Detección del Rostro
- **Detección:**  
  Se utiliza el detector **MTCNN** para analizar la imagen `image_np` y obtener:
  - **boxes:** Coordenadas del cuadro delimitador del rostro.
  - **probs:** Probabilidades de cada detección.
  - **landmarks:** Puntos clave del rostro (por ejemplo, las posiciones de los ojos).

- **Validación:**  
  Si **no se detecta ningún rostro** o si se detectan **múltiples rostros**, se lanza una excepción para garantizar que el proceso solo continúe con una detección única.

### ✂️ Extracción del Rostro
- Se toma el primer cuadro delimitador (`boxes[0]`) y se convierte a enteros.
- Se recorta la región facial de la imagen utilizando las coordenadas del cuadro.
- Se ajustan los **landmarks** restando la posición del cuadro, para que queden relativos a la imagen recortada.

### 🔄 Alineación del Rostro
- Se llama a la función `align_face(face, adjusted_landmarks)` para alinear el rostro basándose en la posición de los ojos.
- Esto corrige la inclinación del rostro, lo cual es crucial para obtener resultados consistentes en el reconocimiento.

### 📐 Redimensionamiento
- Se obtienen las dimensiones (ancho y alto) del rostro alineado.
- Si el **lado menor** de la imagen es mayor a **224 píxeles**, se calcula un factor de redimensionamiento para que dicho lado sea igual a **224 píxeles**, manteniendo la proporción.
- La imagen se redimensiona utilizando `cv2.resize` con interpolación bicúbica (`cv2.INTER_CUBIC`).

### 🔙 Retorno del Rostro Alineado
- La función devuelve la imagen del rostro alineado y redimensionado para ser utilizada en pasos posteriores, como la extracción de **encodings faciales**.

---
### ⚠️ Manejo de Errores
- La función lanza excepciones cuando **no se detecta ningún rostro** o cuando se detectan **múltiples rostros**, facilitando la depuración y garantizando un único rostro a procesar.

### 📏 Redimensionamiento Adecuado
- El ajuste a **224 píxeles** en el lado menor es un estándar en muchos modelos de reconocimiento facial, asegurando consistencia y optimización en el procesamiento.
---

## 📜 Extracción del Encoding Facial

### 🔍  Descripción de la Función  `extract_encoding(aligned_face)`
    Se encarga de extraer el vector de características (encoding) que representa el rostro alineado. Este encoding es fundamental para el reconocimiento facial, ya que permite comparar rostros mediante similitud de vectores.

### 🔍 Detalle del Proceso

### 📥 Entrada:
La función recibe `aligned_face`, la imagen del rostro alineado (se espera que contenga **únicamente el rostro**).

### 📐 Obtención de Dimensiones:
```python
h, w, _ = aligned_face.shape
```
- Se extraen la altura `(h)` y el ancho `(w)` de la imagen. El tercer valor representa los canales de color.

### 📍 Definición de la ubicación del rostro:
```python
face_locations = [(0, w, h, 0)]
```
- Se fuerza a la función de extracción a considerar que la cara ocupa toda la imagen, definiendo la región con:
    - top: 0
    - right: w
    - bottom: h
    - left: 0

### 🧠 Extracción del encoding:
- Se calcula el encoding facial para la región definida. Si no se obtiene ningún encoding, se lanza una excepción.

### 🔙 Retorno:
- Si el encoding se genera correctamente, se retorna el primer encoding (que es el único en este caso).

---
### ⚠️ Manejo de Errores
- La función lanza excepciones cuando **No se pudo generar la codificación facial**, facilitando la depuración.
---
