import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ════════════════════════════════════════════════════════════════
#                   QUESTION 2     
# ════════════════════════════════════════════════════════════════
# Paramètres du paquet d'ondes gaussien initial.
# Remarque : a contrôle la largeur spatiale du paquet.
# Plus a est grand, plus le paquet est étalé en espace
# et localisé en vecteur d'onde (relation d'incertitude).
nx = 1000                
nt = 50000                
t_max = 12.0             
hbar = 1.0

k0  = 1.0               
a = 1.0                 
m = 1.0
x_min, x_max = -60.0, 60.0
largeur_barriere = a

print(f"Paramètres du paquet d'ondes :")
print(f" k0 = {k0}, a = {a}")



# ════════════════════════════════════════════════════════════════
##                      QUESTION 1.a 
# ════════════════════════════════════════════════════════════════


V0 = 1.5

def GaussWP(k0, a, x, t, x0=0.0):
    terme = m * a**2 + 2j * hbar * t
    inv_terme = 1 / terme
    amplitude = np.sqrt(2 * m * a * inv_terme / np.sqrt(2 * np.pi))
    return amplitude \
        * np.exp(inv_terme * m * (a**2 * k0 + 2j * (x - x0))**2 / 4) \
        * np.exp(-a**2 * k0**2 / 4) \
        * np.exp(1j * k0 * x0) 


def Crank_Nicolson(V0, t_max, nt, nx, x_min, x_max, k0, a, x0,largeur_barriere):
    # Méthode de Crank-Nicolson
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(0, t_max, nt)

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    print(f"\nEspace : Δx = {dx:.6f}")
    print(f"Temps : Δt = {dt:.6f}")
    
    # Vérifier stabilité
    stability = hbar * dt / (dx**2)
    print(f"\nCritère de stabilité : {stability:.6f}", end="")
    if stability < 1:
        print(" ✓ STABLE")
    else:
        print(" ✗ INSTABLE - réduire Δt !")

    # Tableau 2D
    psi_0 = GaussWP(k0, a, x, t=0,x0=-10.0)
    psi = np.zeros((nx, nt), dtype=complex)
    psi[:, 0] = psi_0

    V = np.zeros(nx)
    V[(x >= -largeur_barriere/2) & (x <= largeur_barriere/2)] = V0  # barrière en x=0


    r = 1j * hbar * dt / (4 * m * dx**2)
    ##  CREATION DE 2 MATRICES AFIN DE CALCULé LES VALEURS N-1 N ET N+1
    # Contribution du potentiel sur la diagonale
    s = 1j * dt * V / (2 * hbar) 
    # Calcul de N+1 avec A
    A = diags([-r, 1.0 + 2*r + s, -r], [-1, 0, 1], shape=(nx, nx), format='csr') 
    #Caclul de N et N-1 avec B 
    B = diags([r, 1.0 - 2*r - s, r], [-1, 0, 1], shape=(nx, nx), format='csr')

    print(f"\nIntégration (Crank-Nicolson)...")

    for n in range(nt - 1):
        # Calcul de 2 matrices avec l'opérateur @
        psi[:, n+1] = spsolve(A, B @ psi[:, n]) # Fonction qui résout un systeme linéaire afin de trouvé n+1

        #  Limites aux bords
        psi[0, n+1] = 0.0
        psi[-1, n+1] = 0.0

        if n % 100 == 0:
            norm_current = np.sum(np.abs(psi[:, n+1])**2) * dx
            #print(f"n={n}: ||Ψ||² = {norm_current:.6f}")

        if (n + 1) % (nt // 5) == 0:
            print(f"  {(n+1)/nt*100:.0f}% → t = {t[n+1]:.4f}")

    print("✓ Intégration terminée\n")
    return x,t, psi, dx, dt


## Choisir quelle code on veut run pour pas lancer deux code différents

choix = input("Quelle simulation voulez-vous lancer ? (libre / barriere)\n").strip().lower()
if choix == "libre":
    # Simulation libre (pour τ0)
    x, t, psi, dx, dt = Crank_Nicolson(V0=0.0, t_max=t_max, nt=nt, nx=nx,x_min=x_min, x_max=x_max,k0=k0, a=a, x0=-10.0,largeur_barriere=a)
elif choix == "barriere":
    # Simulation avec barrière (pour τt)
    x, t, psi, dx, dt = Crank_Nicolson(V0=1.5, t_max=t_max, nt=nt, nx=nx,x_min=x_min, x_max=x_max,k0=k0, a=a, x0=-10.0,largeur_barriere=a)

# ════════════════════════════════════════════════════════════════
##                      QUESTION 1.b 
# ════════════════════════════════════════════════════════════════

def mesurer_tau0(x, t, psi, a):
    """
    Mesure le temps que met le maximum de |Ψ|² à parcourir une distance a,
    à partir de sa position de départ (simulation libre, V0=0).
    """
    nt = len(t)
    position_max = np.zeros(nt)
    for n in range(nt):
        idx = np.argmax(np.abs(psi[:, n])**2)
        position_max[n] = x[idx]

    x_depart = position_max[0]
    distance_parcourue = position_max - x_depart

    if not np.any(distance_parcourue >= a):
        return None, x_depart

    idx_tau0 = np.argmax(distance_parcourue >= a)
    tau0_num = t[idx_tau0]
    return tau0_num, x_depart

# ════════════════════════════════════════════════════════════════
##                      QUESTION 1.c 
# ════════════════════════════════════════════════════════════════
def mesurer_tau_t(x, t, psi, largeur_barriere):
    """
    Mesure le temps que met une fraction significative de |Ψ|² à franchir
    la barrière, détecté dans la zone transmise (x > largeur_barriere/2).
    """
    nt = len(t)
    zone_transmise = x > (largeur_barriere / 2)

    max_densite_transmise_globale = 0
    for n in range(nt):
        densite_transmise = np.abs(psi[zone_transmise, n])**2
        if np.max(densite_transmise) > max_densite_transmise_globale:
            max_densite_transmise_globale = np.max(densite_transmise)

    seuil = 0.5 * max_densite_transmise_globale

    tau_t_num = None
    for n in range(nt):
        densite_transmise = np.abs(psi[zone_transmise, n])**2
        if np.max(densite_transmise) > seuil:
            tau_t_num = t[n]
            break

    return tau_t_num, max_densite_transmise_globale



# ════════════════════════════════════════════════════════════════
##                      QUESTION 1.d 
# ════════════════════════════════════════════════════════════════

liste_largeurs = [0.5, 1.0, 1.5]

resultats_tau0 = []
resultats_taut = []

for largeur_test in liste_largeurs:
    ## Affiche les parametres
    print(f"  Largeur de barrière testée : {largeur_test}")


    # Simulation libre (V0=0) pour mesurer τ0
    x, t, psi_libre, dx, dt = Crank_Nicolson(V0=0.0, t_max=t_max, nt=nt, nx=nx,x_min=x_min, x_max=x_max,k0=k0, a=a, x0=-10.0,largeur_barriere=largeur_test)
    tau0, x_depart = mesurer_tau0(x, t, psi_libre, largeur_test)
    
    print(f"  Largeur de barrière testée : {largeur_test}")
    
    # Simulation avec barrière (V0=1.5) pour mesurer τt
    x, t, psi_barriere, dx, dt = Crank_Nicolson(V0=1.5, t_max=t_max, nt=nt, nx=nx,x_min=x_min, x_max=x_max,k0=k0, a=a, x0=-10.0,largeur_barriere=largeur_test)
    taut, max_densite = mesurer_tau_t(x, t, psi_barriere, largeur_test)

    resultats_tau0.append(tau0)
    resultats_taut.append(taut)

    print(f"τ0,num = {tau0}")
    print(f"τt,num = {taut}")


# ════════════════════════════════════════════════════════════════
##                      QUESTION 1.e 
# ════════════════════════════════════════════════════════════════

liste_V0 = [0.6, 1.0, 1.5, 2.5, 5.0] 
largeur_fixe = 1.0

resultats_taut_V0 = []

for V0_test in liste_V0:
    print(f"\n{'='*60}")
    print(f"  V0 testé : {V0_test}")
    print(f"{'='*60}")

    x, t, psi_barriere, dx, dt = Crank_Nicolson(V0=V0_test, t_max=t_max, nt=nt, nx=nx,x_min=x_min, x_max=x_max,k0=k0, a=a, x0=-10.0,largeur_barriere=largeur_fixe)
    taut, max_densite = mesurer_tau_t(x, t, psi_barriere, largeur_fixe)
    ratio = max_densite / np.max(np.abs(psi_barriere[:, 0])**2) * 100

    resultats_taut_V0.append(taut)
    print(f"τt,num = {taut}")
    print(f"Ratio de transmission = {ratio:.4f}%")
    ## Le cas ou V0 = 5.0
    if taut is None:
        print("⚠️ Seuil jamais atteint - transmission trop faible pour ce V0")



# VÉRIFICATIONS 

## Vérification énergétique
energy = np.zeros(nt)
for n in range(nt):
    dpsi_dx = np.gradient(psi[:, n], dx)
    energy[n] = np.sum((hbar**2 / (2 * m)) * np.abs(dpsi_dx)**2) * dx


print("Conservation de l'énergie :")
print(f"  E(t=0) = {energy[0]:.6f}")
print(f"  E(t=T) = {energy[-1]:.6f}")
print(f"  Erreur d'énergie = {abs(energy[-1] - energy[0])/energy[0]*100:.4f}%")

## Vérification de la norme
norm = np.array([np.sum(np.abs(psi[:, n])**2) * dx for n in range(nt)])
print("Conservation de la norme :")
print(f"  ||Ψ(t=0)||² = {norm[0]:.6f}")
print(f"  ||Ψ(t=T)||² = {norm[-1]:.6f}")
print(f"  Erreur = {abs(norm[-1] - norm[0])/norm[0]*100:.4f}%")


#  Vérifier si la simulation a divergé
if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
    print("\n❌ ERREUR : La simulation a divergé (NaN ou Inf détectés)")
    print("   Réduisez nt ou augmentez nx")
    exit()

if norm[-1] > norm[0] * 10:
    print("\n⚠️  ATTENTION : La norme a explosé (×10)")
    print("   La simulation est peut-être instable")




# ════════════════════════════════════════════════════════════════
# VISUALISATION (psi)
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Évolution du paquet d'ondes libre (V0=0)", fontsize=14, fontweight='bold')

# Plot 1 — condition initiale
axes[0, 0].plot(x, np.abs(psi[:, 0]), 'b-', linewidth=2)
axes[0, 0].fill_between(x, np.abs(psi[:, 0]), alpha=0.3)
axes[0, 0].set_title("|Ψ(x, t=0)| - Condition initiale")
axes[0, 0].set_xlabel("Position x")
axes[0, 0].set_ylabel("|Ψ|")
axes[0, 0].grid(alpha=0.3)

# Plot 2 — état final
axes[0, 1].plot(x, np.abs(psi[:, -1]), 'r-', linewidth=2)
axes[0, 1].fill_between(x, np.abs(psi[:, -1]), alpha=0.3, color='red')
axes[0, 1].set_title(f"|Ψ(x, t={t_max})| - État final")
axes[0, 1].set_xlabel("Position x")
axes[0, 1].set_ylabel("|Ψ|")
axes[0, 1].grid(alpha=0.3)

# Plot 3 — énergie
axes[1, 0].plot(t, energy, 'g-', linewidth=2)
axes[1, 0].set_title("Énergie cinétique moyenne <E>")
axes[1, 0].set_xlabel("Temps t")
axes[1, 0].set_ylabel("Énergie (U.R.)")
axes[1, 0].set_ylim(energy[0] * 0.9, energy[0] * 1.1)
axes[1, 0].grid(alpha=0.3)

# Plot 4 — heatmap
density = np.abs(psi)**2
vmax = np.max(density)
im = axes[1, 1].imshow(density, aspect='auto', origin='lower',
                       extent=[0, t_max, x_min, x_max], cmap='hot',
                       vmin=0, vmax=vmax, interpolation='bilinear')
axes[1, 1].set_title("|Ψ|² (diagramme x-t)")
axes[1, 1].set_xlabel("Temps t")
axes[1, 1].set_ylabel("Position x")
cbar = plt.colorbar(im, ax=axes[1, 1])
cbar.set_label("|Ψ|²")

plt.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.08, hspace=0.3, wspace=0.3)
plt.savefig("Particule_Libre.png", dpi=150, bbox_inches='tight')
print("✓ Figure sauvegardée : Particule_Libre.png")