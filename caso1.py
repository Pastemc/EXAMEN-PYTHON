#En python: Implementar la segmentacion de una imagen utilizando el algoritmo de k-means clustering
#pasos.
#1.leer una imagen y convertirla a un espacio de color adecuado(por ejemplo, RGB a L*a*b)
#2.aplicar el algoritmo de k-means para segmentar la imagen en k regiones
#3.visualizar la imagen segmentada y comparar los resultados con la imagen original
#Utilizar la libreria de scikit-learn para aplicar k-means clustering
#Acá importamos la libreria de operncv para leer y mostrar 
#la imagen
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Paso 1: Leer la imagen y convertirla a espacio de color L*a*b
image = cv2.imread('imagen.jpg')#ruta de la imagen
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Aplanar la imagen
pixels = image_lab.reshape((-1, 3))

# Paso 2: Aplicar el algoritmo de k-means para segmentar la imagen en k regiones
k = 3  # Número de clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_

# Crear una nueva imagen para la segmentación
segmented_image = np.zeros_like(pixels)

# Asignar a cada pixel el color del cluster correspondiente
for i in range(k):
    segmented_image[labels == i] = np.mean(pixels[labels == i], axis=0)

segmented_image = segmented_image.reshape(image_lab.shape)
segmented_image = segmented_image.astype(np.uint8)

# Convertir de L*a*b a RGB
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB)

# Paso 3: Visualizar la imagen segmentada y comparar los resultados con la imagen original
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image_rgb)
plt.title('Imagen Segmentada')
plt.axis('off')

plt.show()

