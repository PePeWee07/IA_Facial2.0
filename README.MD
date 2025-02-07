# Proyecto de Servidor de Detección de Rostros 

Hemos creado una aplicación utilizando Python con el framework Flask, que tiene la capacidad de detectar rostros en imágenes, recortarlos y cifrarlos para garantizar la seguridad de los datos sensibles.

## 📜Características destacadas

- **Detección de Rostro** 

- **Recorte y Codificación** 

- **Almacenamiento de Codificación** 

## 📎Instrucciones de Uso

1. **Clonar el Repositorio:** Comienza por clonar este repositorio en tu máquina local utilizando el comando:

   ```
   git clone https://github.com/PePeWee07/IA-Facial2.0
   ```

2. **Instalar Dependencias:** Accede al directorio del proyecto e instala las dependencias necesarias `pip install`.

    - torch
    - cmake
    - dlib
    - facenet_pytorch
    - matplotlib
    - face_recognition


3. **En caso de Error con `dlib` seguir los pasos**
    ```
    https://github.com/z-mahmud22/Dlib_Windows_Python3.x
    ```

## 📚 Explicación del Redimensionamiento

### 👉 Condición de Redimensionamiento en el Código
```python
if max(orig_width, orig_height) > max_dim and min(orig_width, orig_height) > 400:
```
Esta condición significa que la imagen **solo se redimensionará si:**
1. **El lado más grande supera 800px** (`max_dim = 800`).
2. **El lado más pequeño supera 400px** (`min(orig_width, orig_height) > 400`).

Esto permite optimizar el procesamiento de imágenes sin afectar la calidad de aquellas que ya tienen un tamaño adecuado.

### 📏 Ejemplo de Funcionamiento en Diferentes Casos
| **Tamaño Original** | **Se Redimensiona?** | **Razón** |
|------------------|---------------|--------|
| **1000x650**    | ✅ Sí         | El ancho (`1000px`) es mayor a `800px`, y la altura (`650px`) es mayor a `400px`. |
| **800x500**      | ✅ Sí         | El ancho (`800px`) cumple con el umbral máximo, y la altura (`500px`) es mayor a `400px`. |
| **800x300**      | ❌ No         | Aunque `800px` cumple el umbral máximo, `300px` es menor a `400px`, por lo que no se toca. |
| **600x400**      | ❌ No         | Ninguno de los lados supera `800px`, así que se mantiene igual. |
| **540x304**      | ❌ No         | La imagen es pequeña, mantenerla evita pérdida de calidad. |

### 👉 ¿Por qué no redimensionamos si el lado menor es menor a 400px?
Si una imagen es **demasiado pequeña** (por ejemplo, `800x300`), es mejor **no cambiar su tamaño**, porque:
✅ **Evita pixelación** → Si ampliamos una imagen con `300px` de altura, perderá calidad.
✅ **Preserva detalles del rostro** → Si la imagen es pequeña, el detector MTCNN ya trabaja con resolución baja, y al modificarla podemos afectar la precisión.

### 🔧 ¿Quieres cambiar esta condición?
Si deseas **siempre** redimensionar si el ancho o el alto supera `800px`, sin importar el tamaño mínimo, puedes modificar la condición en el código así:
```python
if max(orig_width, orig_height) > max_dim:
```
Esto hará que **cualquier imagen con un lado mayor a `800px` se reduzca proporcionalmente**, sin importar el otro lado.

🚀 **Decide según tu caso de uso!**

## 🖼️ Configuración de MTCNN en `facenet-pytorch`

Para optimizar y mejorar la precisión de tu código al utilizar la clase `MTCNN` del paquete `facenet-pytorch`, es importante comprender las configuraciones disponibles y cómo ajustarlas según tus necesidades. A continuación, se detallan los parámetros clave que puedes considerar:

### 🔧 **Parámetros principales**

📏 **`image_size`**: Define el tamaño de las imágenes de salida en píxeles. Por defecto, es `160`. Si las imágenes de entrada son de alta resolución, ajustar este parámetro puede ayudar a mantener la calidad y detalle necesarios para una detección precisa.

🖼️ **`margin`**: Añade un margen al cuadro delimitador en términos de píxeles en la imagen final. Esto es útil para asegurarse de que se capturen áreas adicionales alrededor del rostro, lo que puede ser beneficioso en procesos posteriores como el reconocimiento facial.

📐 **`min_face_size`**: Establece el tamaño mínimo de rostro que el detector buscará. El valor predeterminado es `20`. Si los rostros en las imágenes son más grandes, aumentar este valor puede reducir falsos positivos y mejorar la eficiencia.

🎚️ **`thresholds`**: Son los umbrales de detección para las tres etapas de la red `MTCNN`. Los valores predeterminados son `[0.6, 0.7, 0.7]`. Ajustar estos umbrales puede influir en la sensibilidad y precisión de la detección. Por ejemplo, aumentar los valores puede reducir falsos positivos, pero también podría omitir rostros menos evidentes.

🔍 **`factor`**: Es el factor utilizado para crear una pirámide de escalado de tamaños de rostro. El valor predeterminado es `0.709`. Este parámetro afecta la escala en la que se analizan las imágenes para detectar rostros de diferentes tamaños.

🛠️ **`post_process`**: Indica si se debe postprocesar los tensores de imágenes antes de devolverlos. Por defecto, es `True`. Si estás realizando un procesamiento personalizado después de la detección, es posible que desees desactivar esta opción.

```text
En nuestro caso no necesitamos normalización porque la detección de rostros se usa en un flujo de OpenCV + face_recognition. Si estuviéramos pasando los rostros a un modelo preentrenado en PyTorch, podríamos considerar dpost_process=True
```

🔝 **`select_largest`**: Si es `True`, en caso de que se detecten múltiples rostros, se devolverá el más grande. Si es `False`, se devolverá el rostro con la mayor probabilidad de detección. Esto es útil si esperas que haya múltiples rostros en la imagen y deseas centrarte en uno específico.

🎯 **`selection_method`**: Especifica el método de selección cuando se detectan múltiples rostros. Las opciones incluyen:
  - **`"probability"`**: selecciona el rostro con la mayor probabilidad de detección.
  - **`"largest"`**: selecciona el rostro más grande.
  - **`"largest_over_threshold"`**: selecciona el rostro más grande que supera un cierto umbral de probabilidad.
  - **`"center_weighted_size"`**: selecciona basado en el tamaño de la caja menos el desplazamiento ponderado al cuadrado desde el centro de la imagen.

📂 **`keep_all`**: Si es `True`, se devolverán todos los rostros detectados en el orden dictado por el parámetro `select_largest`. Si se especifica una ruta de guardado (`save_path`), el primer rostro se guardará en esa ruta y los rostros restantes se guardarán con sufijos numéricos.

💻 **`device`**: Especifica el dispositivo en el que se ejecutarán las pasadas de la red neuronal. Los tensores de imágenes y los modelos se copian a este dispositivo antes de ejecutar las pasadas hacia adelante. Por defecto, es `None`, lo que significa que se utilizará la `CPU`, pero si tienes una `GPU` disponible, puedes especificarla para mejorar el rendimiento.

Para una comprensión más profunda y ejemplos prácticos, puedes consultar la guía de `MTCNN` en `facenet-pytorch` y la documentación del [repositorio oficial](https://github.com/timesler/facenet-pytorch).


