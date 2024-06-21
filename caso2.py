#objetivo: detectar y emparejar caracteristicas dentre dos imagenes
#utilizando detectores de caracteristicas (por ejemplo, SIFT o ORB)
#Pasos
#1.Leer dos imagenes que contengan algunos ejemplos comunes.
#2.Detectar las caracteristicas clave en ambas imagenes utilizando
#un detector de caracteristicas.
#3.Describir las caracteristicas detectadas.
#4.Emparejar las caracteristicas entre las dos imagenes.
#5.Visualizar los emparejamientos de caracteristicas en las dos imagenes
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paso 1: Leer las imágenes
img1 = cv2.imread('reategui.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('reategui.jpg', cv2.IMREAD_GRAYSCALE)

# Paso 2: Inicializar el detector y el descriptor (por ejemplo, SIFT)
sift = cv2.SIFT_create()

# Paso 3: Detectar y describir características
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Paso 4: Emparejar características
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Aplicar el test de razón para obtener buenos emparejamientos
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Paso 5: Visualizar los emparejamientos
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostrar la imagen con los emparejamientos usando matplotlib
plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title('Emparejamiento de Características')
plt.axis('off')
plt.show()
