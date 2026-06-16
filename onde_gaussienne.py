import numpy as np
import matplotlib.pyplot as plt

#        QUESTION 2     
nx = 200                # Réduit
nt = 1500                # Augmenté mais pas trop
t_max = 0.3             # Temps plus court

x0  = 0.0
k0  = 1.0               # Réduit (paquet moins énergétique)
sigma = 1.0             # Plus large

print(f"Paramètres du paquet d'ondes :")
print(f"x0 = {x0}, k0 = {k0}, σ = {sigma}")

##       QUESTION 3 
x_min, x_max = -10.0, 10.0
x = np.linspace(x_min, x_max, nx)
t = np.linspace(0, t_max, nt)

dx = x[1] - x[0]
dt = t[1] - t[0]

print(f"\nEspace : Δx = {dx:.6f}")
print(f"Temps : Δt = {dt:.6f}")

# Paquet gaussien
psi_0 = np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Tableau 2D
psi = np.zeros((nx, nt), dtype=complex)
psi[:, 0] = psi_0

    ##  QUESTION 4 
hbar = 1.0
m = 1.0
V0 = 0.0

# Vérifier stabilité
stability = hbar * dt / (dx**2)
print(f"\nCritère de stabilité : {stability:.6f}", end="")
if stability < 1:
    print(" ✓ STABLE")
else:
    print(" ✗ INSTABLE - réduire Δt !")

alpha = -1j * hbar * dt / (2 * m * dx**2)
beta = -1j * V0 * dt / hbar

print(f"α = {alpha}")
print(f" |alpha| = {np.abs(alpha):.6f}")
print(f"β = {beta}")
if np.abs(alpha) > 0.1:
    print("❌ |alpha| trop grand → réduire dt ou augmenter m")

# BOUCLE PRINCIPALE


print(f"\nIntégration...")

absorb_width = 20
absorb = np.ones(nx)
for i in range(absorb_width):
## Eviter que la fonction explose et crée une erreur
    absorb[i] *= (1 - np.cos(np.pi * i / absorb_width)) / 2
    absorb[-(i+1)] *= (1 - np.cos(np.pi * i / absorb_width)) / 2


for n in range(nt - 1):
    for i in range(1, nx - 1):
        laplacian = (psi[i+1, n] - 2*psi[i, n] + psi[i-1, n]) / (dx**2)
        psi[i, n+1] = (psi[i, n] + alpha * laplacian 
                       + 1j * beta * psi[i, n])
    
    psi[:, n+1] *= absorb
    max_val = np.max(np.abs(psi[:, n+1]))

    if max_val > 1e10:
        print(f"⚠️  DIVERGENCE à n={n}, max|Ψ| = {max_val:.2e}")
        print(f"    Réduisez dt ou augmentez nx")
        break
    
    if n % 100 == 0:
        norm_current = np.sum(np.abs(psi[:, n+1])**2) * dx
        print(f"n={n}: ||Ψ||² = {norm_current:.6f}")

    if (n + 1) % (nt // 5) == 0:
        progression = (n + 1) / nt * 100
        print(f"  {progression:.0f}% → t = {t[n+1]:.4f}")

print("✓ Intégration terminée\n")


# VÉRIFICATIONS 


norm = np.array([np.sum(np.abs(psi[:, n])**2) * dx for n in range(nt)])

print("Conservation de la norme :")
print(f"  ||Ψ(t=0)||² = {norm[0]:.6f}")
print(f"  ||Ψ(t=T)||² = {norm[-1]:.6f}")
print(f"  Erreur = {abs(norm[-1] - norm[0])/norm[0]*100:.4f}%")

# ⚠️ Vérifier si la simulation a divergé
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

# Plot 3
axes[1, 0].plot(t, norm, 'g-', linewidth=2)
axes[1, 0].set_title("Conservation de la norme ||Ψ||²")
axes[1, 0].set_xlabel("Temps t")
axes[1, 0].set_ylabel("∫|Ψ|² dx")
axes[1, 0].grid(alpha=0.3)

# Plot 4 - Heatmap (CORRIGÉE)
density = np.abs(psi)**2
# ⚠️ IMPORTANT : Utiliser vmin et vmax pour éviter les problèmes de normalisation
vmax = np.max(density)
im = axes[1, 1].imshow(density, aspect='auto', origin='lower',
                       extent=[0, t_max, x_min, x_max], cmap='hot',
                       vmin=0, vmax=vmax, interpolation='bilinear')
axes[1, 1].set_title("|Ψ|² (diagramme x-t)")
axes[1, 1].set_xlabel("Temps t")
axes[1, 1].set_ylabel("Position x")
cbar = plt.colorbar(im, ax=axes[1, 1])
cbar.set_label("|Ψ|²")

# ⚠️ Utiliser subplots_adjust() au lieu de tight_layout()
plt.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.08, hspace=0.3, wspace=0.3)

plt.show()