import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ════════════════════════════════════════════════════════════════
#                   QUESTION 2     
# ════════════════════════════════════════════════════════════════
# Initialisations des parametres
nx = 200                
nt = 1500                
t_max = 0.3             


k0  = 1.0               
a = 1.0                 
m = 1.0

print(f"Paramètres du paquet d'ondes :")
print(f" k0 = {k0}, a = {a}")


# ════════════════════════════════════════════════════════════════
##                      QUESTION 3 
# ════════════════════════════════════════════════════════════════

x_min, x_max = -10.0, 10.0
x = np.linspace(x_min, x_max, nx)
t = np.linspace(0, t_max, nt)

dx = x[1] - x[0]
dt = t[1] - t[0]

print(f"\nEspace : Δx = {dx:.6f}")
print(f"Temps : Δt = {dt:.6f}")

# Paquet d'onde gaussien
prefactor = (1 / (8 * np.pi**3))**(1/4) * np.sqrt(4 * np.pi * m * a / (m * a**2)) #Préfaceur
# Equation divisé en deux sinon ceci aurait été trop long a écrire
psi_0 = prefactor * np.exp(1j * k0 * x - x**2 / a**2)

# Tableau 2D
psi = np.zeros((nx, nt), dtype=complex)
psi[:, 0] = psi_0


# ════════════════════════════════════════════════════════════════
##                      QUESTION 4 
# ════════════════════════════════════════════════════════════════

hbar = 1.0
V0 = 5.0

# Vérifier stabilité
stability = hbar * dt / (dx**2)
print(f"\nCritère de stabilité : {stability:.6f}", end="")
if stability < 1:
    print(" ✓ STABLE")
else:
    print(" ✗ INSTABLE - réduire Δt !")

# Méthode de Crank-Nicolson

r = 1j * hbar * dt / (4 * m * dx**2)

##  CREATION DE 2 MATRICES AFIN DE CALCULé LES VALEURS N-1 N ET N+1
# Calcul de N+1 avec A
A = diags([-r, 1.0 + 2*r, -r], [-1, 0, 1], shape=(nx, nx), format='csr') 
#Caclul de N et N-1 avec B 
B = diags([r, 1.0 - 2*r, r], [-1, 0, 1], shape=(nx, nx), format='csr')

print(f"\nIntégration (Crank-Nicolson)...")

for n in range(nt - 1):
    # Calcul de 2 matrices avec l'opérateur @
    psi[:, n+1] = spsolve(A, B @ psi[:, n]) # Fonction qui résout un systeme linéaire afin de trouvé n+1

    #  Limites aux bords
    psi[0, n+1] = 0.0
    psi[-1, n+1] = 0.0

    if n % 100 == 0:
        norm_current = np.sum(np.abs(psi[:, n+1])**2) * dx
        print(f"n={n}: ||Ψ||² = {norm_current:.6f}")

    if (n + 1) % (nt // 5) == 0:
        print(f"  {(n+1)/nt*100:.0f}% → t = {t[n+1]:.4f}")

print("✓ Intégration terminée\n")





# VÉRIFICATIONS 

## Verification Energetique
energy = np.zeros(nt)
for n in range(nt):
    # Dérivée spatiale via gradient numérique
    dpsi_dx = np.gradient(psi[:, n], dx)
    # Intégrale de l'énergie cinétique moyenne : ∫ (hbar² / 2m) * |dΨ/dx|² dx
    energy[n] = np.sum((hbar**2 / (2 * m)) * np.abs(dpsi_dx)**2) * dx

print("Conservation de l'énergie :")
print(f"  E(t=0) = {energy[0]:.6f}")
print(f"  E(t=T) = {energy[-1]:.6f}")
print(f"  Erreur d'énergie = {abs(energy[-1] - energy[0])/energy[0]*100:.4f}%")

## Vérification de la Norme 
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
# VISUALISATION
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Évolution du paquet d'ondes gaussien", fontsize=14, fontweight='bold')

# Plot 1
axes[0, 0].plot(x, np.abs(psi_0), 'b-', linewidth=2)
axes[0, 0].fill_between(x, np.abs(psi_0), alpha=0.3)
axes[0, 0].set_title("|Ψ(x, t=0)| - Condition initiale")
axes[0, 0].set_xlabel("Position x")
axes[0, 0].set_ylabel("|Ψ|")
axes[0, 0].grid(alpha=0.3)

# Plot 2
axes[0, 1].plot(x, np.abs(psi[:, -1]), 'r-', linewidth=2)
axes[0, 1].fill_between(x, np.abs(psi[:, -1]), alpha=0.3, color='red')
axes[0, 1].set_title(f"|Ψ(x, t={t_max})| - État final")
axes[0, 1].set_xlabel("Position x")
axes[0, 1].set_ylabel("|Ψ|")
axes[0, 1].grid(alpha=0.3)

# Plot 3 - MODIFIÉ POUR L'ÉNERGIE
axes[1, 0].plot(t, energy, 'g-', linewidth=2)
axes[1, 0].set_title("Énergie totale moyenne <E>")
axes[1, 0].set_xlabel("Temps t")
axes[1, 0].set_ylabel("Énergie (U.R.)")
# Ajustement de l'axe Y pour bien voir que l'énergie est constante (évite une échelle trop aplatie)
axes[1, 0].set_ylim(energy[0] * 0.9, energy[0] * 1.1)
axes[1, 0].grid(alpha=0.3)

# Plot 4 - Heatmap (CORRIGÉE)
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

plt.savefig("onde_gaussienne.png", dpi=150, bbox_inches='tight')
print("✓ Figure sauvegardée : onde_gaussienne.png")
