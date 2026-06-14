"""
Created on Fri May 22 20:10:17 2026

@author: fredrik
"""
import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt

f = 1e-4 
U0 = 1
k = 10 # do not need in reality 


def F(z, phi, k=k):
    arg_z = 4 * (1 + 1j) * phi * np.exp(k * z / 4)
    arg_0 = 4 * (1 + 1j) * phi
    return ((1 + 1j) * f * U0) / (2 * k * phi) * kv(0, arg_z) / kv(1, arg_0)


def surface_angle(phi):
    "Thus is the new Flux one"
    alpha = 4*phi * (1+1j)
    num = kv(0, alpha)
    den = kv(1, alpha)
    ans = (1+1j)*num /  den
    
    theta = np.angle(ans, deg=True)
    return theta

def ekman_transport(phi):
    T = (1j/f) * F(0, phi)
    theta = np.angle(T, deg=True)
    return theta

extent = 4
phis = np.logspace(-50, np.log10(extent), 1000)

#%%
# ===============================
# Plotting
# ===============================
plt.figure(figsize=(8, 5))
plt.suptitle("Surface Angle for Exponetial Model", fontsize=14)
plt.title("Surface angle vs layer thickness",fontsize=13)

angles = surface_angle(phis)

plt.plot(phis, angles)

plt.xlabel(r"Dimensionless layer thickness, $\varphi_\text{exp}$ [-]",fontsize=11)
plt.ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

plt.hlines(45, min(phis), max(phis), color="black", linestyle='--', label="45° reference")
plt.legend(fontsize=11)

plt.ylim(0,95)
plt.yticks([0, 15, 30, 45, 60, 75, 90])
plt.xticks(np.arange(0, extent + 0.5, 0.5))

save_name="exponential_angle"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()


#%%


plt.figure(figsize=(8, 5))
plt.suptitle("Transport Angle for Exponetial Model", fontsize=14)
plt.title("Transport angle vs layer thickness",fontsize=13)

angles = np.array([ekman_transport(phi) for phi in phis])

plt.hlines(135, min(phis), max(phis), color="black", linestyle='--', label="135° reference")
plt.plot(phis, angles)

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Transport angle, $\theta_T$ [deg]",fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

plt.ylim(90,185)
plt.yticks([90, 105, 120, 135, 150, 165, 180])
plt.xticks(np.arange(0, extent + 0.5, 0.5))
plt.legend(fontsize=11)

save_name="exponential_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()

#%%

nu0 = 0.1  # reference viscosity at z=0

ks = [0.02, 0.01]
phis_to_plot = [0.5, 1.0, 2.0]
Nz = 10000

fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex="row", sharey="row")

for i, k_val in enumerate(ks):
    # zmax physically meaningful: where nu has decayed a lot, e.g. 6/k
    zmax = 6 / k_val
    z = np.linspace(0, zmax, Nz)

    ax = axes[0, i]
    for j, phi in enumerate(phis_to_plot):
        F_vals = F(z, phi, k=k_val)
        ax.plot(np.real(F_vals), z, c=f"C{j}", label=fr'$\varphi={phi:.1f}$')
        ax.plot(np.imag(F_vals), z, '--', c=f"C{j}")

    ax.set_xlabel(r"Vertical momentum flux, $F$", fontsize=11)
    ax.set_title(fr"$k = {k_val:.1f}$", fontsize=12)
    ax.set_ylim(0, zmax)
    ax.plot([], [], 'k-',  label=r'$F_x$')
    ax.plot([], [], 'k--', label=r'$F_y$')
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Potential well below ---
    ax_pot = axes[1, i]
    nu_z = nu0 * np.exp(-k_val * z)   # nu(z) = nu0 * exp(-kz)
    ax_pot.plot(1 / nu_z, z, 'k')
    ax_pot.set_xlabel(r"Potential well, $1/\nu$ [m$^{-2}$s]", fontsize=11)
    ax_pot.set_ylim(0, zmax)
    ax_pot.grid(True, linestyle='--', alpha=0.6)
    
axes[0, 0].set_ylabel(r"Height, $z$ [m]", fontsize=11)
axes[1, 0].set_ylabel(r"Height, $z$ [m]", fontsize=11)

plt.suptitle(r"Vertical momentum flux and potential well for exponential viscosity",
             fontsize=14)
plt.tight_layout()
save_name="exponential_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()

