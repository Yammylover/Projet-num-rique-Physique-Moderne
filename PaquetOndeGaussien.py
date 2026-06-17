from scipy.constants import hbar, m_e
import numpy as np
import matplotlib.pyplot as plt

 ## Initialisation des paramètres
m = m_e
k0 = 1e10
a = 1e-9

def GaussWP(k0, a, x, t):
    prefactor = (1 / (8 * np.pi**3))**0.25
    denominator = m * a**2 + 2j * hbar * t
    part_1 = np.sqrt(4 * np.pi * m * a / denominator)
    num_exp = m * (a**2 * k0 + 2j * x)**2
    part_2 = np.exp(m * (a**2 * k0 + 2j * x)**2 / (4 * denominator) - (a**2 * k0**2) / 4)
    return prefactor * part_1 * part_2

x = np.linspace(-5e-9, 5e-9, 1000)
psi = GaussWP(k0, a, x, t=0)

norme = np.max(np.abs(psi))

plt.plot(x, psi.real / norme, label="Partie réelle")
plt.plot(x, psi.imag / norme, label="Partie imaginaire")

plt.legend()
plt.savefig("paquet_ondes.png")