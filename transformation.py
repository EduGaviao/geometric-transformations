#===============================================================================
# Trabalho 4
#-------------------------------------------------------------------------------
# Autor: Eduarda Simonis Gavião
# UNICAMP
#===============================================================================
# Importando bibliotecas
import sys
import numpy as np
import cv2
import math
from PIL import Image
import os
from sympy import frac


#===============================================================================
# TRANSFORMAÇÕES GEOMÉTRICAS (ESCALA)

#Operação de escala 
#recebe as dimensões e o fator de escala

#Interpolação pelo vizinho mais proximo
def escalaNN(img, fator):
    #numero de linhas e de colunas  
    old_h, old_w = img.shape
    #fatores de escala
    h,w = fator
    out= np.empty((h,w),np.uint8) #cria uma matriz formada por zeros (será a imagem de saída)
    w_factor=old_w/w
    h_factor=old_h/h

    for i in range(h):
        for j in range(w):
            out[i,j]=img[int(i*h_factor),int(j*w_factor)]  #aplica a transformação
    filename='escalaNearestNeighbors.png' #salva a imagem 
    cv2.imwrite(filename,out.astype(np.uint8))
    return out.astype(np.uint8)


#Interpolação bilinear
def escalaBL(img,fator):
    #numero de linhas e de colunas
    old_h, old_w = img.shape
    #fatores de escala
    h,w = fator
    out= np.empty((round(h),round(w)),np.uint8) #cria uma matriz formada por zeros (será a imagem de saída)
    w_factor=old_w/w
    h_factor=old_h/h

    for i in range(round(h)):
        for j in range(round(w)):
            #aplica a transformação
            #mapeia as coordenas para a imagem original
            x = i*h_factor 
            y = j*w_factor
            # verifica as coordenada dos quatro pixels vizinhos
            x_f= math.floor(x)
            x_c=min(old_h-1,math.ceil(x))
            y_f = math.floor(y)
            y_c=min(old_w-1,math.ceil(y))
            
            #controlando hipóteses em quem o valor 0 é passado
            if (x_c == x_f) and (y_c==y_f):
                value=img[int(x),int(y)]
            elif (x_c == x_f):    
                v1=img[int(x),int(y_f)]
                v2=img[int(x),int(y_c)]
                value= v1*(y_c-y)+v2*(y-y_f)
            elif (y_c == y_f):    
                v1=img[int(x_f),int(y)]
                v2=img[int(x_c),int(y)]
                value= v1*(x_c-x)+v2*(x-x_f)
            else:    
                #obtendo os valores dos pixels vizinhos
                p1= img[x_f,y_f]
                p2= img[x_c,y_f]
                p3= img[x_f,y_c]
                p4= img[x_c,y_c]
                #estimando os valores com base nos vizinhos
                v1=p1*(x_c-x)+p2*(x-x_f)
                v2=p3*(x_c-x)+p4*(x-x_f)
                value= v1*(y_c-y)+v2*(y-y_f)
            #repassa o valor para coordenada correta na imagem de saída    
            out[i,j]=value
    filename='escalaBilinear.png'
    cv2.imwrite(filename,out.astype(np.uint8))
    return out.astype(np.uint8)

## Etipulando o kernel baseado na equação de interpolação bicubica
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0
  
  
# função de preenchimento 
def padding(img, H, W):
    zimg = np.zeros((H+4, W+4))
    zimg[2:H+2, 2:W+2] = img
      
    # controle para as bordas
    zimg[2:H+2, 0:2] = img[:, 0:1]
    zimg[H+2:H+4, 2:W+2] = img[H-1:H, :]
    zimg[2:H+2, W+2:W+4] = img[:, W-1:W]
    zimg[0:2, 2:W+2] = img[0:1]
      
    # controle para os pontos faltantes
    zimg[0:2, 0:2] = img[0, 0]
    zimg[H+2:H+4, 0:2] = img[H-1, 0]
    zimg[H+2:H+4, W+2:W+4] = img[H-1, W-1]
    zimg[0:2, W+2:W+4] = img[0, W-1]
    return zimg
  
#Interpolação pelo vizinho mais proximo
def rotacaoNN(img, angulo):
    #numero de linhas e de colunas  
    old_h, old_w = img.shape
    
    out= np.empty((old_h,old_w),np.uint8) #cria uma matriz formada por zeros (será a imagem de saída)
    for i in range(old_h):
        for j in range(old_w): 
            x = round(i* math.cos(math.radians(angulo)) + j * math.sin(math.radians(angulo)))
            y = round(-i* math.sin(math.radians(angulo)) + j * math.cos(math.radians(angulo))) 
            try:
                if int(x) < 0 or int(y) < 0:
                    raise Exception
                out[i][j] = img[int(x)][int(y)]
            except:
                out[i][j] = 0
    filename='rotacaoNearestNeighbors.png' #salva a imagem 
    cv2.imwrite(filename,out.astype(np.uint8))
    return out.astype(np.uint8)  
# aplicação da tranformação bicubica
def bicubic(img, scale):
     #numero de linhas e de colunas
    old_h, old_w = img.shape
    img = padding(img, old_h, old_w)
    #fatores de escala
    dH = math.floor(old_h*scale)
    dW = math.floor(old_w*scale)
    out= np.empty((dH,dW),np.uint8) #cria uma matriz formada por zeros (será a imagem de saída)
    
    h=1/scale

    #imagem é percorrida 
    for i in range(dH):
        for j in range(dW):
            x= i*h
            y= j*h
            dx=x%1
            dy=y%1
            #controla pontos fora da img
            if x> old_h or x<0:
                out[i, j]=0
            elif y> old_w or y<0:
                out[i, j]=0
            else:
                out[i,j]=biFunction(img,dx,dy,int(x),int(y))

    filename='escalaBicubica.png' #salva a imagem 
    cv2.imwrite(filename,out.astype(np.uint8))
    return out.astype(np.uint8)


def escalaLG(img,fator):
    #numero de linhas e de colunas
    old_h, old_w = img.shape
    img = padding(img, old_h, old_w)
    #fatores de escala
    dH = math.floor(old_h*fator)
    dW = math.floor(old_w*fator)
    out= np.empty((dH,dW),np.uint8) #cria uma matriz formada por zeros (será a imagem de saída)
    
    h=1/fator

    #imagem é percorrida 
    for i in range(dH):
        for j in range(dW):
            x= i*h
            y= j*h
            dx=x%1
            dy=y%1
            
            #o polinômio é dividido 
            L1=((-dx*(dx-1)*(dx-2)*img[int(x)-1, int(y)+1-2])/6)
            L2=(((dx+1)*(dx-1)*(dx-2)*img[int(x), int(y)+1-2])/2)
            L3=((-dx*(dx+1)*(dx-2)*img[int(x)+1, int(y)+1-2])/2)
            L4=((dx*(dx+1)*(dx-1)*img[int(x)+2, int(y)+1-2])/6)

            P1=L1+L2+L3+L4
            out1=(P1*(-dy*(dy-1)*(dy-2)))/6

            L1=(-dx*(dx-1)*(dx-2)*img[int(x)-1, int(y)+2-2])/6
            L2=((dx+1)*(dx-1)*(dx-2)*img[int(x), int(y)+2-2])/2
            L3=(-dx*(dx+1)*(dx-2)*img[int(x)+1, int(y)+2-2])/2
            L4=(dx*(dx+1)*(dx-1)*img[int(x)+2, int(y)+2-2])/6

            P2=L1+L2+L3+L4
            out2=(P2*((dy+1)*(dy-1)*(dy-2)))/2

            L1=(-dx*(dx-1)*(dx-2)*img[int(x)-1, int(y)+3-2])/6
            L2=((dx+1)*(dx-1)*(dx-2)*img[int(x), int(y)+3-2])/2
            L3=(-dx*(dx+1)*(dx-2)*img[int(x)+1, int(y)+3-2])/2
            L4=(dx*(dx+1)*(dx-1)*img[int(x)+2, int(y)+3-2])/6

            P3=L1+L2+L3+L4
            out3=(P3*(-dy*(dy+1)*(dy-2)))/2

            L1=(-dx*(dx-1)*(dx-2)*img[int(x)-1, int(y)+4-2])/6
            L2=((dx+1)*(dx-1)*(dx-2)*img[int(x), int(y)+4-2])/2
            L3=(-dx*(dx+1)*(dx-2)*img[int(x)+1, int(y)+4-2])/2
            L4=(dx*(dx+1)*(dx-1)*img[int(x)+2, int(y)+4-2])/6

            #por fim a intensidade final é encotrada por meio das aplicações dos
            #coeficientes L e a soma dos fatores restantes
            P4=L1+L2+L3+L4
            out4=(P4*(dy*(dy+1)*(dy-1)))/6

            out[i, j]=out1+out2+out3+out4
    filename='escalaLagrange.png' #salva a imagem 
    cv2.imwrite(filename,out.astype(np.uint8))
    return out.astype(np.uint8)


#===============================================================================
# # TRANSFORMAÇÕES GEOMÉTRICAS (Rotação)

#Operação de rotação
#recebe as dimensões e o angulo

#Interpolação pelo vizinho mais proximo
def rotacaoBL(img, angulo):
    old_h, old_w = img.shape #tamanho da imagem anterior
    
    out= np.empty((old_h,old_w),np.uint8) #nova imagem
    cosseno= math.cos(math.radians(angulo))
    seno= math.sin(math.radians(angulo))

    for i in range(old_h):
        for j in range(old_w):
            #passando o centro das coordenadas
            #rotação
            x = i* cosseno + j * seno
            y = -i* seno + j * cosseno 

            x0= math.floor(x)
            x1=min(old_h-1,math.ceil(x))
            y0 = math.floor(y)
            y1=min(old_w-1,math.ceil(y))
            
            x0 = np.clip(x0, 0, img.shape[1]-1);
            x1 = np.clip(x1, 0, img.shape[1]-1);
            y0 = np.clip(y0, 0, img.shape[0]-1);
            y1 = np.clip(y1, 0, img.shape[0]-1);

            Ia = img[ x0, y0 ]
            Ic = img[ x1, y0 ]
            Ib = img[ x0, y1 ]
            Id = img[ x1, y1 ]


            wa = (x1-x) * (y1-y)
            wb = (x1-x) * (y-y0)
            wc = (x-x0) * (y1-y)
            wd = (x-x0) * (y-y0)

            value=wa*Ia + wb*Ib + wc*Ic + wd*Id
            #repassa o valor para coordenada correta na imagem de saída    
            out[i,j]=value
            
    filename='rotacaoBilinear.png' #salva a imagem 
    cv2.imwrite(filename,out.astype(np.uint8))
    return out.astype(np.uint8)                

def biFunction(img,dx,dy,x,y):
    old_h, old_w = img.shape
    #declarando os indices da série
    m=np.arange(-1, 3, 1)
    n=np.arange(-1, 3, 1)
    f=0
    #fazendo laços para contabilizar o somatório
    for i in m:
        for j in n:
            #definindo a função R(s)
            #primeiro para x
            Rm=(i-dx)
            px1= Rm+2 if Rm+2>0 else 0 #trata a condição de t>0 = t e t<0 0
            px2= Rm+1 if Rm+1>0 else 0 #trata a condição de t>0 = t e t<0 0
            px3= Rm if Rm>0 else 0 #trata a condição de t>0 = t e t<0 0
            px4= Rm-1 if Rm-1>0 else 0 #trata a condição de t>0 = t e t<0 0
            # Descobrindo R
            Rx= (1/6)*((px1**3)-4*(px2**3)+6*(px3**3)-4*(px4**3))
            #agora para y
            Rn=(dy-j)
            py1= Rn+2 if Rn+2>0 else 0 #trata a condição de t>0 = t e t<0 0
            py2= Rn+1 if Rn+1>0 else 0 #trata a condição de t>0 = t e t<0 0
            py3= Rn if Rn>0 else 0 #trata a condição de t>0 = t e t<0 0
            py4= Rn-1 if Rn-1>0 else 0 #trata a condição de t>0 = t e t<0 0
            # Descobrindo R
            Ry= (1/6)*((py1**3)-4*((py2)**3)+6*((py3)**3)-4*((py4)**3))

            #controla pontos fora da img
            if (i+x)>= old_h or (i+x)<0:
                 f=0
            elif (j+y)>= old_w or (j+y)<0:
                f=0
            else:#juntando os processos para descobrir a intensidade
                f= f + (img[i+x,j+y]*Rx*Ry)
    
    return f
#Interpolação pelo vizinho mais proximo
def rotacaoBC(img, angulo):
    
    old_h, old_w = img.shape
    out= np.empty((old_h,old_w),np.uint8) #cria uma matriz formada por zeros (será a imagem de saída)

    for i in range(old_h):
        for j in range(old_w):
            x = round(i* math.cos(math.radians(angulo)) + j * math.sin(math.radians(angulo)))
            y = round(-i* math.sin(math.radians(angulo)) + j * math.cos(math.radians(angulo)))  
            dx=x%1
            dy=y%1

            #controla pontos fora da img
            if x> old_h or x<0:
                out[i, j]=0
            elif y> old_w or y<0:
                out[i, j]=0
            else:
                out[i,j]=biFunction(img,dx,dy,int(x),int(y))
        
    filename='rotacaoBicubica.png' #salva a imagem 
    cv2.imwrite(filename,out.astype(np.uint8))
    return out.astype(np.uint8) 


#Interpolação pelo vizinho mais proximo
def rotacaoLG(img, angulo):
    old_h, old_w = img.shape
    img = padding(img, old_h, old_w)
    #fatores de escala

    out= np.empty((old_h,old_w),np.uint8) #cria uma matriz formada por zeros (será a imagem de saída)
    
    #percorre a imagem
    for i in range(old_h):
        for j in range(old_w):
            x = round(i* math.cos(math.radians(angulo)) + j * math.sin(math.radians(angulo)))
            y = round(-i* math.sin(math.radians(angulo)) + j * math.cos(math.radians(angulo))) 
            dx=x%1
            dy=y%1
            #controla pontos fora da img
            if x> old_h or x<0:
                out[i, j]=0
            elif y> old_w or y<0:
                out[i, j]=0
            else:
                #o polinômio é dividido 
                L1=((-dx*(dx-1)*(dx-2)*img[int(x)-1, int(y)+1-2])/6)
                L2=(((dx+1)*(dx-1)*(dx-2)*img[int(x), int(y)+1-2])/2)
                L3=((-dx*(dx+1)*(dx-2)*img[int(x)+1, int(y)+1-2])/2)
                L4=((dx*(dx+1)*(dx-1)*img[int(x)+2, int(y)+1-2])/6)
    
                P1=L1+L2+L3+L4
                out1=(P1*(-dy*(dy-1)*(dy-2)))/6
    
                L1=(-dx*(dx-1)*(dx-2)*img[int(x)-1, int(y)+2-2])/6
                L2=((dx+1)*(dx-1)*(dx-2)*img[int(x), int(y)+2-2])/2
                L3=(-dx*(dx+1)*(dx-2)*img[int(x)+1, int(y)+2-2])/2
                L4=(dx*(dx+1)*(dx-1)*img[int(x)+2, int(y)+2-2])/6
    
                P2=L1+L2+L3+L4
                out2=(P2*((dy+1)*(dy-1)*(dy-2)))/2
    
                L1=(-dx*(dx-1)*(dx-2)*img[int(x)-1, int(y)+3-2])/6
                L2=((dx+1)*(dx-1)*(dx-2)*img[int(x), int(y)+3-2])/2
                L3=(-dx*(dx+1)*(dx-2)*img[int(x)+1, int(y)+3-2])/2
                L4=(dx*(dx+1)*(dx-1)*img[int(x)+2, int(y)+3-2])/6
    
                P3=L1+L2+L3+L4
                out3=(P3*(-dy*(dy+1)*(dy-2)))/2
    
                L1=(-dx*(dx-1)*(dx-2)*img[int(x)-1, int(y)+4-2])/6
                L2=((dx+1)*(dx-1)*(dx-2)*img[int(x), int(y)+4-2])/2
                L3=(-dx*(dx+1)*(dx-2)*img[int(x)+1, int(y)+4-2])/2
                L4=(dx*(dx+1)*(dx-1)*img[int(x)+2, int(y)+4-2])/6
                #por fim a intensidade final é encotrada por meio das aplicações dos
                #coeficientes L e a soma dos fatores restantes    
                P4=L1+L2+L3+L4
                out4=(P4*(dy*(dy+1)*(dy-1)))/6
    
                out[i, j]=out1+out2+out3+out4
    filename='rotacaoLagrange.png' #salva a imagem  
    cv2.imwrite(filename,out.astype(np.uint8))
    return out.astype(np.uint8)
            
    
     
#===============================================================================
def main ():
    print('Escolha um dos métodos de Transformação:')
    print('Transformação Geometrica- Escala - 1')
    print('Transformação Geometrica- Rotação - 2')
    print('Sair - 0')
    op = input("Indique a operação ")

    #tratamento de opções
    if op =="1":
        os.system('cls')
        print('Escolha um dos métodos de Interpolação:')
        print('Interpolação pelo Vizinho Mais Próximo - 1')
        print('Interpolação Bilinear - 2')
        print('Interpolação Bicúbica - 3')
        print('Interpolação Lagrange - 4')
        interpolacao=input("Indique a operação: ")
        
        if interpolacao == "1":

            print("Deseja trabalhar com:")
            print("Fator de Escala - 1")
            print("Dimensões finais da imagem - 2")
            esc = input("Indique a operação: ")

            if esc == "1":
                name= input("Digite o nome da imagem que deseja manipular:")
                img= Image.open(name + '.png')
                img = np.array(img)   

                if img is None:
                    print ('Erro abrindo a imagem.\n')
                    sys.exit ()

                print("Em relação as dimenções da imagem final: ")
                e= float(input("Digite o fator de escala "))
                
                old_h, old_w = img.shape
                new_h= round(old_h*e)
                new_w= round(old_w*e)
                fator = (new_h, new_w)
                    
                out= escalaNN(img,fator)
            if esc == "2": 
                name= input("Digite o nome da imagem que deseja manipular:")
                img= Image.open(name + '.png')
                img = np.array(img)   

                if img is None:
                    print ('Erro abrindo a imagem.\n')
                    sys.exit ()   

                print("Em relação as dimenções da imagem final: ")
                R= int(input("Digite o número de Linhas: "))
                C= int(input("Digite o número de Colunas: "))

                fator = (R, C)

                out= escalaNN(img,fator)

            cv2.imshow("Transformacao de Escala Vizinho mais Proximo",out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if interpolacao == "2":

            print("Deseja trabalhar com:")
            print("Fator de Escala - 1")
            print("Dimensões finais da imagem - 2")
            esc = input("Indique a operação: ")

            if esc == "1":
                name= input("Digite o nome da imagem que deseja manipular:")
                img= Image.open(name + '.png')
                img = np.array(img)   

                if img is None:
                    print ('Erro abrindo a imagem.\n')
                    sys.exit ()

                print("Em relação as dimenções da imagem final: ")
                e= float(input("Digite o fator de escala "))
                
                old_h, old_w = img.shape
                new_h= old_h*e
                new_w= old_w*e
                fator = (new_h, new_w)
                
                
                out= escalaBL(img,fator)
            if esc == "2": 
                name= input("Digite o nome da imagem que deseja manipular:")
                img= Image.open(name + '.png')
                img = np.array(img)   

                if img is None:
                    print ('Erro abrindo a imagem.\n')
                    sys.exit ()   

                print("Em relação as dimenções da imagem final: ")
                R= int(input("Digite o número de Linhas: "))
                C= int(input("Digite o número de Colunas: "))

                fator = (R, C)

                out= escalaBL(img,fator)

            cv2.imshow("Trasnformacao de Escala Bilinear",out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if interpolacao == "3":

            name= input("Digite o nome da imagem que deseja manipular:")
            img= Image.open(name + '.png')
            img = np.array(img)   
            if img is None:
                print ('Erro abrindo a imagem.\n')
                sys.exit ()
            print("Em relação as dimenções da imagem final: ")
            e= float(input("Digite o fator de escala "))
            out=bicubic(img,e)


            cv2.imshow("Trasnformacao de Escala Bilinear",out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if interpolacao == "4":
            name= input("Digite o nome da imagem que deseja manipular:")
            img= Image.open(name + '.png')
            img = np.array(img)   

            if img is None:
                print ('Erro abrindo a imagem.\n')
                sys.exit ()

            print("Em relação as dimenções da imagem final: ")
            e= float(input("Digite o fator de escala "))
                
            out= escalaLG(img,e)
            
    
            cv2.imshow("Trasnformacao de Escala Interpolação Lagrange",out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif op == "2":
        os.system('cls')
        print('Escolha um dos métodos de Interpolação:')
        print('Interpolação pelo Vizinho Mais Próximo - 1')
        print('Interpolação Bilinear - 2')
        print('Interpolação Bicúbica - 3')
        print('Interpolação Lagrange - 4')
        interpolacao=input("Indique a operação: ")
        
        if interpolacao == "1":
            name= input("Digite o nome da imagem que deseja manipular:")
            img= Image.open(name + '.png')
            img = np.array(img)   

            if img is None:
                print ('Erro abrindo a imagem.\n')
                sys.exit ()

            print("Em relação a rotacao da imagem final: ")
            angle= float(input("Digite o angulo de rotação "))
                
            out= rotacaoNN(img,angle)
            cv2.imshow("Transformacao de Rotação Interpolação Vizinho mais proximo",out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        if interpolacao == "2":
            name= input("Digite o nome da imagem que deseja manipular:")
            img= Image.open(name + '.png')
            img = np.array(img)   

            if img is None:
                print ('Erro abrindo a imagem.\n')
                sys.exit ()

            print("Em relação a rotacao da imagem final: ")
            angle= float(input("Digite o angulo de rotação "))
                
            out= rotacaoBL(img,angle)

            cv2.imshow("Transformacao de Rotação Interpolação bilinear",out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            
        if interpolacao == "3":

            name= input("Digite o nome da imagem que deseja manipular:")
            img= Image.open(name + '.png')
            img = np.array(img)   

            if img is None:
                print ('Erro abrindo a imagem.\n')
                sys.exit ()

            print("Em relação a rotacao da imagem final: ")
            angle= float(input("Digite o angulo de rotação "))
                
            out= rotacaoBC(img,angle)
            cv2.imshow("Transformacao de Rotação Interpolação Bicúbica",out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if interpolacao =="4":
            name= input("Digite o nome da imagem que deseja manipular:")
            img= Image.open(name + '.png')
            img = np.array(img)   

            if img is None:
                print ('Erro abrindo a imagem.\n')
                sys.exit ()

            print("Em relação a rotacao da imagem final: ")
            angle= float(input("Digite o angulo de rotação "))
                
            out= rotacaoLG(img,angle)
            cv2.imshow("Transformacao de Rotação Interpolação por polinômio de lagrange",out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
       


    else: 
        print('Execução Finalizada')


if __name__ == '__main__':
    main()

