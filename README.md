# Rúbrica IA – Corte 3 – Parte 1

## Clasificación supervisada de radiografías de tórax con CNN

Este proyecto implementa un **modelo de Deep Learning supervisado** para la **clasificación binaria** de radiografías de tórax en dos categorías: **Normal** y **Neumonía**. Se utiliza una **Red Neuronal Convolucional (CNN)** construida desde cero y entrenada sobre un dataset público ampliamente usado en investigación.

El trabajo corresponde a la **Parte 1 de la rúbrica del Curso 3 de Inteligencia Artificial**, enfocada en aprendizaje supervisado.

---

## Objetivo

Diseñar, entrenar y evaluar un modelo CNN capaz de **clasificar imágenes médicas de rayos X de tórax**, aplicando un flujo completo de aprendizaje supervisado que incluya:

* Preprocesamiento de datos
* Entrenamiento del modelo
* Evaluación cuantitativa
* Análisis visual de resultados

---

## Metodología

1. **Descarga del dataset** desde Kaggle: *Chest X-Ray Pneumonia*.
2. **Preprocesamiento de imágenes**:

   * Conversión a escala de grises.
   * Redimensionamiento a 64×64 píxeles.
   * Normalización de valores de píxel (0–1).
3. **Etiquetado de datos**:

   * 0 → Normal
   * 1 → Neumonía
4. **División del dataset** en conjuntos de entrenamiento y prueba (80/20).
5. **Diseño de la CNN**:

   * Capas convolucionales y pooling
   * Capa fully connected
   * Dropout para reducción de overfitting
6. **Entrenamiento del modelo**.
7. **Evaluación** mediante accuracy, matriz de confusión y métricas de clasificación.
8. **Visualización** de predicciones correctas e incorrectas.

---

##  Arquitectura del modelo CNN

* Conv2D (16 filtros, 3×3) + MaxPooling
* Conv2D (32 filtros, 3×3) + MaxPooling
* Flatten
* Dense (64 neuronas, ReLU)
* Dropout (0.3)
* Dense (1 neurona, Sigmoid)

Función de pérdida: **Binary Crossentropy**
Optimizador: **Adam**

---

##  Métricas de evaluación

* **Accuracy** en conjunto de prueba
* **Matriz de confusión**
* **Precision, Recall y F1-score** (classification report)
* Gráficas de **accuracy** y **loss** durante el entrenamiento

---

##  Estructura del proyecto

```
Rubrica Parte 1/
│
├── cnn_clasificacion_parte1.py
├── requirements.txt
├── .gitignore
├── README.md
```

>  El dataset y las credenciales de Kaggle **no se incluyen** en el repositorio por buenas prácticas.

---

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/USUARIO/NOMBRE_REPO.git
cd NOMBRE_REPO
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar la API de Kaggle (`kaggle.json`) según la documentación oficial.

---

##  Ejecución

```bash
python cnn_clasificacion_parte1.py
```

El script descargará el dataset (si no existe), entrenará el modelo y mostrará:

* Curvas de entrenamiento
* Métricas de evaluación
* Matriz de confusión
* Ejemplos de predicciones correctas e incorrectas

---

##  Tecnologías utilizadas

* Python 3
* TensorFlow / Keras
* OpenCV
* Scikit-learn
* NumPy
* Matplotlib

---

##  Contexto académico

* **Curso:** Inteligencia Artificial – Corte 3
* **Actividad:** Rúbrica – Parte 1
* **Tipo de aprendizaje:** Supervisado
* **Dataset:** Chest X-Ray Pneumonia (Kaggle)
* **Autor:** Camilo Salazar

---

##  Conclusión

Este proyecto demuestra la aplicación práctica de **redes neuronales convolucionales** para la clasificación de imágenes médicas, siguiendo un enfoque supervisado completo y alineado con los objetivos académicos del curso.

Sirve como base comparativa frente a enfoques no supervisados desarrollados en etapas posteriores del trabajo.
