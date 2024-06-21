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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb

def kmeans_segmentation(image_path, k=3):
    # Paso 1: Leer la imagen y convertirla al espacio de color L*a*b
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = rgb2lab(image_rgb)

    # Obtener las dimensiones de la imagen
    w, h, d = image_lab.shape

    # Reshape the image to a 2D array of pixels
    image_array = image_lab.reshape((w * h, d))

    # Paso 2: Aplicar el algoritmo de k-means para segmentar la imagen en k regiones
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image_array)
    segmented_labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Crear la imagen segmentada
    segmented_image_lab = centers[segmented_labels].reshape(image_lab.shape)

    # Convertir la imagen segmentada de L*a*b a RGB
    segmented_image_rgb = lab2rgb(segmented_image_lab)

    return image_rgb, segmented_image_rgb

def main():
    # Ruta de la imagen
    image_path = 'imagen.jpg'  # Reemplaza con la ruta a tu imagen

    # Número de clusters
    k = 3

    # Paso 3: Segmentar la imagen y visualizar los resultados
    original_image, segmented_image = kmeans_segmentation(image_path, k)

    # Mostrar la imagen original y la imagen segmentada
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f'Segmentacion en k clusters')
    plt.axis('off')

    plt.show()

if _name_ == "_main_":
    main()        


