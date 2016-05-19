import numpy as np
import cv2
import random
import operator,math
from matplotlib import pyplot as plt
#Primera Solucion Aleatoria
def primeraSolucion(n,RangoMenor,RangoMayor):
	rango=[]
	for j in range(0,n):
		rango.append(random.uniform(RangoMenor,RangoMayor))
	return rango
#Evaluarfx
def evaluarFx(vector,datos):
	fx = 0.0
	m=1
	#Aptitud Problema 1
	for x in range(0,len(datos)):
		h0=vector[0]+(vector[1]*x)
		r1 = h0*x
		r2 = pow((r1-datos[x]),2)
		if(x<39):
			m=((datos[x+1]-datos[x])/((x+1)-x))
		if m==0:
			m=1
		#print x
		#print m
		#fx=1
		s=1/(2*m)
		fx=s*r2
	return fx

#Generar 10 hijos por padre y ver el mejor
def generarHijos(papa,sigma,datos):
	hijo=[]
	exitos=0
	iteracionMejor=0
	for i in range(0,10):
		hijo=generarHijo(papa,sigma)
		fxP=evaluarFx(papa,datos)
		fxh=evaluarFx(hijo,datos)
		better=Comparar(fxP,fxh)
		if(i% (2*(len(papa))) == 0):
			sigma=ModificarSigma(exitos,sigma,len(papa))
			exitos=0
		if(better):
			#print "cambio"
			exitos=exitos+1
			#print exitos
			#imprimir((i+1),padre,hijos,fxP,fxh,exitos,hijos)
			papa=hijo
			iteracionMejor=i
		else:
			#print "no"
			papa=papa
	return papa,iteracionMejor,sigma

#generar hijo Mutar
def generarHijo(papa,sigma):
	hijo=[]
	for j in range (0,len(papa)):
		hijo.append(papa[j]+(sigma*(random.uniform(-1,1))))
	return hijo

#Comparar al padre con el hijo
def Comparar(Fxpadre,FXhijo):
	if (Fxpadre <= FXhijo):
		return False
	else:
		return True
#MadificarSIgma
def ModificarSigma(exitos,sigma,d):
	ps=exitos/(10*d)
	if(ps==.20):
		return (sigma)
	if (ps < .20):
		return (sigma * 0.817)
	if (ps > .20):
		return (sigma * 0.817)

#programa principal donde se lleva acavo laestrategia evolutiva
#recive NoDeProblema,LimiteInferior,LimeteSuperior,No.Variables
def menor(RangoMenor,RangoMayor,d,s,datos):
	sigma=s
	MAXGEN=1000
	#10000
	exitos=0
	iteracionMejor=0
	padre = primeraSolucion(d,RangoMenor,RangoMayor)
	for i in range (0,MAXGEN):
		padre,mejori,sigma=generarHijos(padre,sigma,datos)
		#if(i% (10*d) == 0):
		#	sigma=ModificarSigma(exitos,sigma,d)
		#	exitos=0
		#print mejori
		if (mejori!=0):
			iteracionMejor=i
			exitos=mejori
	fxP=evaluarFx(padre,datos)
	#print hijos
	#print "Mejor:",padre
	#print "Mejor Aptitud:",fxP
	#print "No.iteracion",iteracionMejor
	#print "NO.Subiteracion",exitos
	return padre

#-----------------------------------------------------------------------------------------------------------------
def predecir(padre):
	img = cv2.imread('IMG_130.JPG')
	img=caraCuadro(img,100,5)
	#resul=evaluarFx(img,img2)
	t=padre[0]+(padre[1]*41)
	#print t
	hx=(1/(1+(math.exp(t))))
	predic=(0.0001)*(pow((hx-img),2))
	#print img
	print predic
	print padre
	return predic

#--------------------------------------------------------------------------------------------------------------
def caraCuadro(img,n,p):
	#print "entro"
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	#print faces
	for (x,y,w,h) in faces:
		Rec = img[y:y+h, x:x+w]
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
	his=histograma(Rec)
	nom=str(p)+str(n)+".jpg"
	cv2.imwrite(nom,Rec);
	return sumar(his)

def leerc():
    global A
    archivo = open("numeros.txt", "r") 
    contenido = archivo.read()
    A=contenido.split(' ')
 
def escribir(c,nombre):
    ResTxt = open(nombre,"a")
    for i in range (len(c)):
        ResTxt.write(str(c[i]))
        ResTxt.write(' ')
    ResTxt.close()
def encontar(predic,datos):
	n=0
	if(predic<13000):
		n=2
	else:
		n=1
	#for i in range(0,len(datos)):
	#	if(predic<=datos[i]):
	#		n=n
	#	else:
	#		n=n+1
	return n

#def evaluarFx(im1,im2):
#	rms= cv2.compareHist(im1,im2,3)
	#rms = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b)**2, im1, im2))/len(im1))
	#return rms

def histograma(img):
	#hist=img.ravel
	#histr=plt.hist(img.ravel(),256,[0,256]);
	color = ('b','g','r')
	for i,col in enumerate(color):
   		histr = cv2.calcHist([img],[i],None,[256],[0,256])
   		cv2.normalize(histr,histr,0,255,cv2.NORM_MINMAX)
	return histr

def main():
	datos=[]
	preim='IMG_13' #jorge uno
	datos=Entrenar(preim,1)
	preim='IMG_14' #omar dos
	datos=datos+Entrenar(preim,2)
	#escribir(datos,"histo.txt")
	#Costo(datos)
	print datos
	padre=menor(-1,1,2,0.1,datos)
	predic=predecir(padre)
	n=encontar(predic,datos)
	imagenencontrada='IMG_'+str(n)+'.JPG'
	print "La persona es la numero:"+ str(n)
	#comparar(datos)

def Entrenar(preim,p):
	persona=[]
	#preim=IMG_14
	for i in range (0,20):
		imagen=preim+str(i)+'.JPG'
		#print imagen
		img = cv2.imread(imagen)
		persona.append(caraCuadro(img,i,p))
	return persona

def Costo(datos):
	print datos
	print datos.size

def sumar(img):
	#print type(lap)
	#print type(lap[0][0][0])
	#punto=img.sum()
	return img.sum()

#def compara(datos):
#	img = cv2.imread('IMG_135.JPG')
#	dato=caraCuadro(img,100)
#	print dato
#	for i in range (0,2):
#		for j in range (0,20):
#			d=datos[i][j]
#			print datos[i][j],i,j
#			if (d==dato):
#				print "si"
#			else:
#				print "no"

main()
#prueba()