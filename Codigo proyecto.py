import cv2

import numpy as np
from plantcv import plantcv as pcv
import skimage
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2

from skimage.util import img_as_float

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#Se carga una imagen de prueba


base=cv2.imread("carro0.jpg")




#Se pasa a hsv y se separan las placas

hsv=cv2.cvtColor(base, cv2.COLOR_BGR2HSV)


mask_amarillo = cv2.inRange(hsv, (10,50,50), (30,255,130))





#Se erosiona y dilata la mascara para eliminar "impurezas" 

estructura=cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5,5))




mask_amarillo = cv2.morphologyEx(mask_amarillo, cv2.MORPH_OPEN, estructura)


mask_amarillo=cv2.dilate(mask_amarillo, estructura, iterations=3)



#Se aplica la mascara a la imagen 

amarillo = cv2.bitwise_and(base, base, mask=mask_amarillo)





cv2.imshow("normal",base)

cv2.imshow("placas",amarillo)

cv2.waitKey(0)
cv2.destroyAllWindows()






#En las imagenes a utilizar el interes son las placas que aparecen en la zona
#de abajo, por lo que se corta el resto

dimensiones=amarillo.shape

alto=int(dimensiones[0]/2)
ancho=int(dimensiones[1]/2)


interes=amarillo[alto-200:dimensiones[0],:,:]

cv2.imshow("zona_interes",interes)

cv2.waitKey(0)
cv2.destroyAllWindows()





#Se pasa la zona de interes a escala de grises y se hace un umbral para 
#operaciones posteriores


gris=interes[:,:,0]

n,gris = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY)





#Al usar el umbral algunas zonas dentro de la placa no quedan blancas, se 
#dilata y se erosiona la imagen para contrarestrar esto


estructura=cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (9,9))

gris=cv2.morphologyEx(gris, cv2.MORPH_CLOSE, estructura)


cv2.imshow("zona_gris",gris)

cv2.waitKey(0)
cv2.destroyAllWindows()


#Como ya se tienen en blanco las zonas de las placas se procedera a identificar
#su posición, para esto se encuentran sus bordes primero

contorno,k=cv2.findContours(gris,cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)





cont=1



izquierdo=0
derecho=0




for i in contorno:
        
#Se recorre cada contorno y se descarta si no es lo suficientemente grande
#o si es demasiado grande

            if cv2.contourArea(i)>3000 and cv2.contourArea(i)<6500:
            
#Si el contorno es lo suficientemente grande significa que es de una placa
#completa, se identifican las coordenadas de la esquina izquierda del
#rectangulo más pequeño que la cubre, ademas de su altura y grosor

                x,y,grosor,alto = cv2.boundingRect(i)



#Se corta la zona especifica en donde esta ubicada cada placa y se almacena
                
                if cont==1 and x<=ancho:
                    
                    placaizquierda=interes[y:y+alto,x:x+grosor,:]
                    cont=2
                    izquierdo=1
                    
                    
                    
                if cont==1 and x>ancho:
                    
                    placaderecha=interes[y:y+alto,x:x+grosor,:]
                    cont=2
                    derecho=1
                    
                    
                
                if cont==2:
                    placaderecha=interes[y:y+alto,x:x+grosor,:]
                    derecho=1
                    
                    
                    
#Se dibujan los rectangulos en la imagen para verificar que se este haciendo
#bien
          
                # interes=cv2.rectangle(interes,(x,y),(x+grosor,y+alto),
                #           (0,0,255),3)
                
               





#En el caso de que no se hallan encontrado 2 placas se revisa por si hay
#placas blancas

#En el caso de que no se hallan encontrado 2 placas se revisa por si hay
#placas blancas



if izquierdo==0 or derecho==0:
    
    
    
    blancog=cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    
    
    
    
    #amarillogris=cv2.subtract(amarillogris,150)
    
    
    mask_blanco, m = pcv.threshold.custom_range(img=blancog, lower_thresh=[80], upper_thresh=[200], channel='gray')






    #Se erosiona y dilata la mascara para eliminar "impurezas" 

    estructura=cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5,5))




    mask_blanco = cv2.morphologyEx(mask_blanco, cv2.MORPH_OPEN, estructura)


    mask_blanco=cv2.dilate(mask_blanco, estructura, iterations=3)



    #Se aplica la mascara a la imagen 

    blanco = cv2.bitwise_and(base, base, mask=mask_blanco)





    cv2.imshow("normal",base)

    cv2.imshow("placasblanco",blanco)

    cv2.waitKey(0)
    cv2.destroyAllWindows()






    #En las imagenes a utilizar el interes son las placas que aparecen en la zona
    #de abajo, por lo que se corta el resto

    dimensiones=blanco.shape

    alto=int(dimensiones[0]/2)
    ancho=int(dimensiones[1]/2)


    interes=blanco[alto-80:dimensiones[0],:,:]

    cv2.imshow("zona_interes",interes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





    #Se pasa la zona de interes a escala de grises y se hace un umbral para 
    #operaciones posteriores


    gris=interes[:,:,0]

    n,gris = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY)





    #Al usar el umbral algunas zonas dentro de la placa no quedan blancas, se 
    #dilata y se erosiona la imagen para contrarestrar esto


    estructura=cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5,5))

    gris=cv2.morphologyEx(gris, cv2.MORPH_OPEN, estructura)
    
    #gris=cv2.erode(gris, estructura, iterations=1)


    cv2.imshow("zona_gris",gris)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #Como ya se tienen en blanco las zonas de las placas se procedera a identificar
    #su posición, para esto se encuentran sus bordes primero

    contorno,k=cv2.findContours(gris,cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    
    
    for i in contorno:
            
    #Se recorre cada contorno y se descarta si no es lo suficientemente grande
    #o si es demasiado grande
                

                if cv2.contourArea(i)>400 and cv2.contourArea(i)<5000:
                
    #Si el contorno es lo suficientemente grande significa que es de una placa
    #completa, se identifican las coordenadas de la esquina izquierda del
    #rectangulo más pequeño que la cubre, ademas de su altura y grosor

                    x,y,grosor,alto = cv2.boundingRect(i)

                    
                    
                    if grosor>1.2*alto and grosor<2.5*alto:
                        
                        
                        
    #Se corta la zona especifica en donde esta ubicada cada placa y se almacena
                    
                        if cont==1 and x<=ancho:
                            
                            placaizquierdab=interes[y:y+alto,x:x+grosor,:]
                            cont=2
                            izquierdo=1
                            
                            
                            
                        if cont==1 and x>ancho:
                            
                            
                            
                            placaderechab=interes[y:y+alto+10,x:x+grosor,:]
                            cont=2
                            derecho=1
                            
                            
                        
                        if cont==2 and derecho==0:
                            placaderechab=interes[y:y+alto+10,x:x+grosor,:]
                            derecho=1
                            
                        
                        if cont==2 and izquierdo==0:
                            placaizquierdab=interes[y:y+alto+10,x:x+grosor,:]
                            izquierdo=1



                 



if izquierdo==1:
    cv2.imshow("Placa izquierda",placaizquierda)

if derecho==1:
    cv2.imshow("Placa derecha",placaderecha)


if derecho==1 or izquierdo==1:
    cv2.waitKey(0)
    cv2.destroyAllWindows()









#Se cambia la perspectiva en la imagen, metodo tomado de [2]



# Se determina el cambio de las coordenadas (parametros determinados por prueba
#y error)

if izquierdo==1:
    dimensiones1=placaizquierda.shape
    alto=dimensiones1[0]
    ancho=dimensiones1[1]
    
    input1 = np.float32([[5,10], [ancho-5,8], [ancho-5,alto-11], [5,alto-10]])
    output1 = np.float32([[0,0], [ancho-1,0], [ancho-1,alto-1], [0,alto-1]])
    
    
    
    # se genera la matriz de perspectiva y se procesa la imagen
    
    matrix = cv2.getPerspectiveTransform(input1,output1)
    
    placaizquierda = cv2.warpPerspective(placaizquierda, matrix, (ancho,alto), cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0,0,0))



# Se determina el cambio de las coordenadas (parametros determinados por prueba
#y error) pero para la placa 2


if derecho==1:
    dimensiones2=placaderecha.shape
    alto=dimensiones2[0]
    ancho=dimensiones2[1]
    
    input1 = np.float32([[5,10], [ancho-5,8], [ancho-5,alto-11], [5,alto-10]])
    output1 = np.float32([[0,0], [ancho-1,0], [ancho-1,alto-1], [0,alto-1]])
    
    
    
    # se genera la matriz de perspectiva y se procesa la imagen para la placa 2
    
    matrix = cv2.getPerspectiveTransform(input1,output1)
    
    placaderecha = cv2.warpPerspective(placaderecha, matrix, (ancho,alto), cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0,0,0))




if izquierdo==1:
    cv2.imwrite("placaizq.jpg",placaizquierda)
    placaizquierda = cv2.detailEnhance(placaizquierda, sigma_s=10, sigma_r=0.15)
    cv2.imshow("Placa izquierda",placaizquierda)
    
if derecho==1:
    cv2.imwrite("placader.jpg",placaderecha)
    placaderecha = cv2.detailEnhance(placaderecha, sigma_s=2, sigma_r=0.5)
    cv2.imshow("Placa derecha",placaderecha)
    

if derecho==1 or izquierdo==1:
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    

# img1=cv2.imread("placader.jpg",0)

# cv2.imshow("Salida1",img1)


# create trackbars for color chang
def nothing(x):
    pass
def filtro_ocr(img1):
    
    cv2.namedWindow('result')
    cv2.createTrackbar('d1','result',1,255, nothing)
        
    norm_img = np.zeros((img1.shape[0], img1.shape[1]))
    img1 = cv2.normalize(img1, norm_img, 0, 255, cv2.NORM_MINMAX)
    
    blur = cv2.GaussianBlur(img1,(5,5),0)
    cv2.imshow("blur",blur)
    edges = cv2.Canny(img1,100,100)
    
    cv2.imshow("edges",edges)
    
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("th3",th3)
    ret2,th2 = cv2.threshold(blur,35,255,cv2.THRESH_BINARY)
    
    cv2.imshow("th2",th2)
    
     

    # =============================================================================
    # while(True):
    #    d1 = cv2.getTrackbarPos('d1','result')
    #   
    #    cv2.imshow("result",th2) 
    #    k = cv2.waitKey(1) & 0xFF
    #    if k == ord('q'):
    #       break
    # =============================================================================
    
    

    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.dilate(th2, kernel, iterations = 1)

    #sal = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("erosion",erosion)
    r1=pytesseract.image_to_string(th2,config='--psm 11')
    r2=pytesseract.image_to_string(th3,config='--psm 11')
    r3=pytesseract.image_to_string(erosion,config='--psm 11')
    r4=pytesseract.image_to_string(img1,config='--psm 11')
    
    if (len(r1)<=9) and len(r1)>6:
        print("resultadoOCR R1",r1)
    if len(r2)<=9 and len(r2)>6:
        print("resultadoOCR R2",r2)
    if len(r3)<=9 and len(r3)>6:
        print("resultadoOCR R3",r3)
    if len(r4)<=9 and len(r4)>6:
        print("resultadoOCR R4",r4)
    # print(len(r2))
    # ret2,th2 = cv2.threshold(blur,80,255,cv2.THRESH_BINARY)
# =============================================================================
#     
#     print("resultadoOCR ",pytesseract.image_to_string(th2,config='--psm 11'))
#     print("resultadoOCR ",pytesseract.image_to_string(th3,config='--psm 11'))
#     print("resultadoOCR ",pytesseract.image_to_string(erosion,config='--psm 11'))
#     print("resultadoOCR ",pytesseract.image_to_string(img1,config='--psm 11'))
#     
# =============================================================================
    cv2.waitKey(0) 
      
    cv2.destroyAllWindows() 
    

if izquierdo==1:
    img1=cv2.imread("placaizq.jpg",0)
    print("resultado placa izq:")
    filtro_ocr(img1)
    
    
if derecho==1:
    img2=cv2.imread("placader.jpg",0)
    print("resultado placa der:")
    filtro_ocr(img2)
    
    
    
#Se pasan las placas a escala de grises

# =============================================================================
# 
# if izquierdo==1:    
#     placaizquierda_g=cv2.cvtColor(placaizquierda, cv2.COLOR_BGR2GRAY)
#     placa_izq=pytesseract.image_to_string(placaizquierda_g)
#     print("las placas del vehiculo de la izquierda son: ",placa_izq)
# if derecho==1:
#     placaderecha_g=cv2.cvtColor(placaderecha, cv2.COLOR_BGR2GRAY)
#     placa_der=pytesseract.image_to_string(placaderecha_g)
#     print("las placas del vehiculo de la derecha son: ",placa_der)
# 
# 
# =============================================================================




# if izquierdo==1:
#     cv2.imshow("Placa izquierda gris",placaizquierda_g)

# if derecho==1:
#     cv2.imshow("Placa derecha gris",placaderecha_g)


# if derecho==1 or izquierdo==1:
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()







# =============================================================================
# 
# #Se equaliza el histograma de las placas
# 
# clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# 
# if izquierdo==1:
#     placaizquierda_g = clahe.apply(placaizquierda_g)
#     
# if derecho==1:    
#     placaderecha_g = clahe.apply(placaderecha_g)
# 
# 
# 
# if izquierdo==1:
#     cv2.imshow("Placa izquierda ecualizada",placaizquierda_g)
# 
# if derecho==1:
#     cv2.imshow("Placa derecha ecualizada",placaderecha_g)
# 
# 
# if derecho==1 or izquierdo==1:
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# 
# 
# 
# =============================================================================




#Se aplica el filtro wiener para reducir ruido por movimiento 
#(se añade algo de ruido intencional antes para poder lograrlo), parametros
#definidos por prueba y error (metodo tomado de la referencia [1])



# =============================================================================
# 
# if izquierdo==1:
#     placaizquierda_g = img_as_float(placaizquierda_g)
#     
# if derecho==1:
#     placaderecha_g = img_as_float(placaderecha_g)
# 
# 
# 
# psf = np.ones((2, 2)) / 25
# rng = np.random.default_rng()
# 
# 
# if izquierdo==1:
#     placaizquierda_g = conv2(placaizquierda_g, psf)
#     
# if derecho==1:
#     placaderecha_g = conv2(placaderecha_g, psf)
# 
# 
# 
# 
# if izquierdo==1:
#     placaizquierda_g += 0.1 * placaizquierda_g.std() * rng.standard_normal(placaizquierda_g.shape)
# 
# 
# if derecho==1:
#     placaderecha_g += 0.1 * placaderecha_g.std() * rng.standard_normal(placaderecha_g.shape)
# 
# 
# if izquierdo==1:
#     claro1 = skimage.restoration.unsupervised_wiener(placaizquierda_g, psf)
#     # cv2.imwrite('palOCR.png',claro1[0])
# 
# if derecho==1:
#     claro2 = skimage.restoration.unsupervised_wiener(placaderecha_g, psf)
#     # cv2.imwrite('palOCR.png',claro2[0])
# 
# 
# 
# fp=cv2.imread('palOCR.png')
# 
# if izquierdo==1:   
#     cv2.imshow("procesado1",fp)
#     placa_izq=pytesseract.image_to_string(claro1[0])
#     print("las placas del vehiculo de la izquierda son: ",placaizquierda_g)
# if derecho==1:
#     cv2.imshow("procesado2",fp)
#     placa_der=pytesseract.image_to_string(claro2[0])
#     print("las placas del vehiculo de la derecha son: ",placaderecha_g)
# 
# if derecho==1 or izquierdo==1:
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# 
# =============================================================================




cv2.waitKey(0)
cv2.destroyAllWindows()










#Referencias

#[1]https://scikit-image.org/docs/dev/auto_examples/filters/plot_restoration.html
#[2]https://stackoverflow.com/questions/63954772/perspective-transform-in-opencv-python