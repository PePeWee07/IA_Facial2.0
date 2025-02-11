# 📋 Registro de Personas en MongoDB

Este documento explica el proceso completo para registrar una persona en la base de datos utilizando el procesamiento facial. El script realiza las siguientes tareas:

- **Carga y preprocesamiento de la imagen:**  
  Se carga la imagen, se preprocesa, se detecta el rostro, se alinea y se redimensiona para optimizar la extracción del encoding facial.

- **Extracción del encoding facial:**  
  Se extrae el vector de características (encoding) que representa el rostro.

- **Guardado del rostro procesado:**  
  La imagen del rostro alineado se guarda en una ruta definida en el sistema.

- **Registro en MongoDB:**  
  Se inserta un documento en la base de datos que contiene el ID de la cara, la ruta de la imagen, el encoding facial y el nombre de la persona.

---
