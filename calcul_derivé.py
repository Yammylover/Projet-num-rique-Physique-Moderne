import numpy

def square(x): #swuare function
    return x*x

def double(x):  #2x function
    return 2*x

def pourcentageerreur(val_th,val_algo):
    value=((val_algo-val_th)/val_th)*100
    return absolute(value)

def absolute(x):
    if x < 0:
        return x*-1
    else:
        return x

#calculer la dérivée de x²

size=100   #quantité de valeurs utilisées: + haut, + précis
tab_x = numpy.linspace(0,10,size)  #initialiser 100 valeurs de x, entre 0 et 10
tab_f = square(tab_x)   #possible en num.py

#print(tab_f)
#print(tab_x)

tab_th=[0,2,6,8,10,12,14,16,18,20]
tab_result=[]

#pas des x
h = tab_x[1]-tab_x[0]
print("h=",h)

value=0
for i in range(size-1):
    if i < size :
        value = (tab_f[i+1]-tab_f[i])/(h)#calcul de la valeur de la dérivée en tab_x[i]
        theorie = double(tab_x[i])      #calcul de la valeur à partir de la fonction connue
        tab_result.append(value)        #ajout de value au tableau des results
        #print ("dérivé de f en ",tab_x[i]," = ",i,"est",value)
        print("dérivé theorique",theorie,"dérivé de l'algorithme",value)    #comparaison des deux valeurs
        print("pourcentage d'erreur:",pourcentageerreur(theorie,value))     #calcul du pourcentage d'erreur
        print("écart:",absolute(value-theorie))     #calcul du pourcentage d'erreur
        #if value != double(tab_x[i]):
        #    print("Différent!!!!!!")
    else:
        print("error")


# print(tab_result[i])