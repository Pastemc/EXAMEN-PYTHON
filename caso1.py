
#En python: Implementar la segmentacion de una imagen utilizando el algoritmo de k-means clustering
#pasos.
#1.leer una imagen y convertirla a un espacio de color adecuado(por ejemplo, RGB a L*a*b)
#2.aplicar el algoritmo de k-means para segmentar la imagen en k regiones
#3.visualizar la imagen segmentada y comparar los resultados con la imagen original

#Acá importamos la libreria de operncv para leer y mostrar 
#la imagen
import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_segmentation(image_path, k=3):
    #K=k-means y es como un medidor para cambiar al k-means
    # Paso 1: Leer la imagen y convertirla al espacio de color L*a*b
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)  
    pixel_values = image_lab.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Paso 2: Aplicar el algoritmo de k-means para segmentar la imagen en k regiones
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_lab.shape)
    # Convertimos de L*a*b a RGB
    segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_Lab2RGB)

    return image_rgb, segmented_image_rgb

def main():
    # Llamaamos a la imagen
    image_path = 'ara.jpg'  # Reemplaza con la ruta a tu imagen

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
    plt.title(f'Segmentacion en k Clusters')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()



