import numpy as np

def square(x):
    return x*x

def double_prime(x):
    return 2*np.ones_like(x) # Création d'un tableau de 2 de la meme taille que x

def pourcentage_erreur(val_theo,val_calcule):
    if val_theo == 0:
        return abs(val_theo) #afin d'eviter la division par 0
    return abs((val_calcule - val_theo)/val_theo) *100

size = 100
tab_x = np.linspace(0,10,size)
tab_f = square(tab_x)
h = tab_x[1] - tab_x[0]

print(f"h = {h:6f}") # Affiche la valeur de h


for i in range(1, size - 1):
    d2_algo = (tab_f[i+1] - 2*tab_f[i] + tab_f[i-1]) / (h**2)
    d2_th   = 2.0  

    print(f"x = {tab_x[i]:.4f} | f''(algo) = {d2_algo:.6f} | f''(théorie) = {d2_th:.6f} | erreur = {pourcentage_erreur(d2_th, d2_algo):.4f}%")