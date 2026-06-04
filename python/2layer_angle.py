"""
Created on Fri May 22 20:10:17 2026

@author: fredrik
"""
import numpy as np
import matplotlib.pyplot as plt


# Parameters
f = 1e-4  # Coriolis parameter
U0 = 1.0  # Reference velocity

def layer_dUdz0(phi, epsilon):
      phi = phi/np.sqrt(2)
      R = (1 - epsilon) / (1 + epsilon)
      A1 = np.exp(phi)
      E1 = A1 * (np.cos(phi) + 1j * np.sin(phi))
      Z = (1 + 1j) * (E1**2 - R) / (E1**2 + R)
      return Z
  
def U(z_tilde, phi, epsilon):
    "This solution is normalizd for h_1"

    gamma = epsilon
    den = 1 - gamma + np.exp(np.sqrt(2)*(1 + 1j)*phi) * (1 + gamma)
    A1 = -U0 * ((1 - gamma) / den)
    B1 = -U0 * ((1 + gamma) * np.exp(np.sqrt(2)*(1 + 1j)*phi) / den)
    B2 = -U0 * (2* np.exp(((1 + 1j)/np.sqrt(2)) * phi * (1 + 1/gamma))/ den)

    U1 =  U0 + A1 * np.exp((1 + 1j)*z_tilde/np.sqrt(2)) + B1 * np.exp(-(1 + 1j)*z_tilde/np.sqrt(2))
    U2 =  U0 + B2 * np.exp(-(1 + 1j)*z_tilde/(np.sqrt(2)*gamma))

    return np.where(z_tilde < phi, U1, U2)


def ekman_transport(phi, epsilon, zmax=1500.0, Nz=500000):
    z = np.linspace(0, zmax, Nz)
    U_vals = U(z, phi, epsilon)
    M = np.trapezoid(U_vals-U0, z)
    print(np.angle(M, deg=True), np.angle(M, deg=True)-np.angle(layer_dUdz0(phi, epsilon), deg=True))
    return M


# Plottin gparameters
extent = 5

nu_ratios = np.array([0.00001, 0.1, 0.5, 1/0.5, 1/0.1, 1/0.00001])
epsilons = np.sqrt(nu_ratios)

phis = np.linspace(0,extent,100000)

#%%
plt.figure(figsize=(8*extent/3,5))
for epsilon in epsilons:

    layer_angle = np.angle(layer_dUdz0(phis, epsilon), deg=True)

    plt.plot(phis, layer_angle, label=fr'$\nu_2/\nu_1=={epsilon**2:.0e}$')


plt.hlines(45, min(phis), max(phis), color="black", linestyle='--', label="45° reference")
plt.vlines([np.pi/np.sqrt(2), np.sqrt(2)*np.pi], 5, 90, color="black", linestyle=":", label=r"$\pi/\sqrt{2}$ and $\sqrt{2}\pi$")

plt.xlabel(r"Dimensionless lower layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
plt.suptitle("Surface Angle for 2-layer Model", fontsize=14)
plt.title("Surface angle vs lower layer thickness",fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="lower right", fontsize=11)
plt.ylim(0,95)
plt.yticks([0,15,30,45,60,75,90])
plt.xticks(np.arange(0, extent + 0.5, 0.5))
save_name="2layer_angle"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

#%%
phis = np.linspace(0,extent,300)
nu_ratios = np.array([0.00001, 0.1, 0.5, 1/0.5, 1/0.1, 1/0.00001])
epsilons = np.sqrt(nu_ratios)

plt.figure(figsize=(8*extent/3,5))
for epsilon in epsilons:

    M_vals = np.array([ekman_transport(phi, epsilon) for phi in phis])
    trans_angle = np.angle(M_vals, deg=True)
    plt.plot(phis, trans_angle, label=fr'$\nu_2/\nu_1={epsilon**2:.0e}$')

# -------------------------
# Reference lines
# -------------------------
plt.vlines([np.pi/np.sqrt(2), np.sqrt(2)*np.pi], 90, 185, color="black", linestyle=":", label=r"$\pi/\sqrt{2}$ and $\sqrt{2}\pi$")

plt.xlabel(r"Dimensionless lower layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Transport angle, $\theta_T$ [deg]", fontsize=11)
plt.suptitle("Transport Angle for 2-layer Model", fontsize=14)
plt.title("Transport angle vs lower layer thickness",fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="lower right", fontsize=11)
plt.ylim(90,185)
plt.yticks([90, 105, 120, 135, 150, 165, 180])
plt.xticks(np.arange(0, extent + 0.5, 0.5))
save_name="2layer_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

