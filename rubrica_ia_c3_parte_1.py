
"""
Clasificación supervisada de radiografías de tórax
usando una CNN desde cero.

Dataset: Chest X-Ray Pneumonia (Kaggle)
Curso: IA - CORTE 3
Actividad: Rúbrica Parte 1
"""

#Aqui intalamos Kaglles y opendatasets obtenemos el dataset publico con el que vamos a trabajar
!pip install kaggle
!pip install opendatasets

import opendatasets as od
if not os.path.exists("chest-xray-pneumonia"):
    od.download("https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
    print("Dataset descargado.")
else:
    print("Dataset ya existe.")


#Importamos las librerias con que vamos a trabajar y procesamos los datos
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = "chest-xray-pneumonia/chest_xray"

#Aqui se hace la carga, preprocesamiento y  se etiqueta los datos para el modelo
def load_images_from_folder(folder, label, limit=1200):
    images = []
    labels = []
    count = 0

    for filename in os.listdir(folder):
        if count >= limit:
            break
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(label)
            count += 1

    return images, labels

X = []
y = []


#En esta parte se realiza la carga de los datos y el preprocesamiento antes de que los datos sean pasados al modelo CNN
normal_train, normal_train_lab = load_images_from_folder(DATA_PATH + "/train/NORMAL", 0, 600)
pneu_train, pneu_train_lab = load_images_from_folder(DATA_PATH + "/train/PNEUMONIA", 1, 600)

normal_test, normal_test_lab = load_images_from_folder(DATA_PATH + "/test/NORMAL", 0, 300)
pneu_test, pneu_test_lab = load_images_from_folder(DATA_PATH + "/test/PNEUMONIA", 1, 300)

X = normal_train + pneu_train + normal_test + pneu_test
y = normal_train_lab + pneu_train_lab + normal_test_lab + pneu_test_lab

X = np.array(X).reshape(-1, 64, 64, 1) / 255.0
y = np.array(y)

#Separamos los datos de entrenamiento de de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Dataset listo!")
print("X shape:", X.shape)
print("Train:", X_train.shape)
print("Test:", X_test.shape)

#Unas imagenes de ejemplo
import matplotlib.pyplot as plt

#Con Plt para visualizarlas
plt.figure(figsize=(8,4))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.title("Normal" if y[i]==0 else "Neumonía")
    plt.axis("off")
plt.show()

#Este es el modelo CNN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D((2,2)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

#Entrenamiento
hist = model.fit(
    X_train, y_train,
    epochs=8,
    batch_size=32,
    validation_split=0.2
)

#Eavalucion del modelo
cnn_loss, cnn_acc = model.evaluate(X_test, y_test)
print("")
print(f"Precisión del modelo CNN: {cnn_acc:.4f}")

#Grafica para mostrar el Acurracy
print("")
plt.figure(figsize=(12,4))
print("")

# Accuracy
plt.subplot(1,2,1)
plt.plot(hist.history['accuracy'], label='Entrenamiento')
plt.plot(hist.history['val_accuracy'], label='Validación')
plt.title("Accuracy durante entrenamiento")
plt.legend()

#Perdidas del proceso
plt.subplot(1,2,2)
plt.plot(hist.history['loss'], label='Entrenamiento')
plt.plot(hist.history['val_loss'], label='Validación')
plt.title("Pérdida durante entrenamiento")
plt.legend()

plt.show()

#Como plus damos una matriz de confucion
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = (model.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)

print(" ")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión")
plt.show()
print(" ")

#Implementamos algunas metricas
from sklearn.metrics import classification_report

print("Metricas")
print(classification_report(y_test, y_pred, target_names=["Normal", "Neumonía"]))
print(" ")

#Visualuzamos algunas imagenes predrictas
y_scores = model.predict(X_test).flatten()

correctos = np.where(y_test == y_pred.reshape(-1))[0]
incorrectos = np.where(y_test != y_pred.reshape(-1))[0]

print(f"Correctos: {len(correctos)} | Incorrectos: {len(incorrectos)}")

m_correctos = correctos[:3]
m_incorrectos = incorrectos[:3]

plt.figure(figsize=(12, 8))

#Correctos
for i, idx in enumerate(m_correctos):
    img = X_test[idx].squeeze()
    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap="gray")
    real = "Neumonía" if y_test[idx] == 1 else "Normal"
    pred = "Neumonía" if y_pred[idx] == 1 else "Normal"
    plt.title(f" Correcto\nR:{real} | P:{pred}\nScore:{y_scores[idx]:.2f}")
    plt.axis("off")

#Incorrectos
for i, idx in enumerate(m_incorrectos):
    img = X_test[idx].squeeze()
    plt.subplot(2, 3, i+4)
    plt.imshow(img, cmap="gray")
    real = "Neumonía" if y_test[idx] == 1 else "Normal"
    pred = "Neumonía" if y_pred[idx] == 1 else "Normal"
    plt.title(f" Incorrecto\nR:{real} | P:{pred}\nScore:{y_scores[idx]:.2f}")
    plt.axis("off")

plt.tight_layout()
plt.show()