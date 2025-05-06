# Proyecto-Final-Procesamiento-de-Imágenes
﻿## **Resumen del proyecto**

**Objetivo**
Segmentar placas de vehiculos motorizados mediante técnicas de procesamiento de imagenes no basadas en IA, para posteriormente identificar sus caracteres mediante un OCR.

**Muestras utilizadas**
Se utilizaron 15 imagenes diferentes tomadas desde la misma cámara de seguridad, las imágenes podian contener características diversas, como tener solo un vehiculo en vez de dos, placas de colores no convencionales, etc.

**Metodos utilizados**
Para la segmentación, se utilizó umbralización por color, closing, segmentación por ubicación, analisis de particulas, umbralización en escala de grises, transformación de perspectiva, filtrado (Gaussiano y Wiener), ecualización de histograma y binarización.

**Librerias utilizadas**
Cv2, Numpy, Plantcv, skimage, matplotlib, scipy, pytesseract.

**Resultados**
Se obtuvo una precisión promedio de 0,83 para el conjunto de prueba, detalles en el informe pdf adjunto.

