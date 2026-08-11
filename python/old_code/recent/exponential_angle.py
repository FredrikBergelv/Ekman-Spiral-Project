"""
Created on Fri May 22 20:10:17 2026

@author: fredrik
"""
import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt

f = 1e-4 
U0 = 10
k = 10 # do not need in reality 


def tau(z, phi, k=k):
    arg_z = 2 * (1 + 1j) * phi * np.exp(k * z / 2)
    arg_0 = 2 * (1 + 1j) * phi
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
    T = (1j/f) * tau(0, phi)
    theta = np.angle(T, deg=True)
    return theta

extent = 4
phis = np.linspace(1e-50, extent, 1000)



#%%
# ===============================
# Plotting
# ===============================
plt.figure(figsize=(8, 5))
plt.suptitle("Surface Angle for Exponetial Model", fontsize=14)

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
hEk = np.sqrt(2 * nu0 / f)

phis_to_plot = [0.2, 1.0, 4.0]
Nz = 98
zmax = 5 *hEk
z = np.linspace(0, zmax, Nz)


fig, axes = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex="row", sharey="row")

ax = axes[0]
for j, phi in enumerate(phis_to_plot):
        
        k_val = 1/(hEk*phi)
        tau_vals = tau(z, phi, k=k_val)
        ang = np.angle(tau_vals, deg=True)
        #ax.plot(ang, z, c=f"C{j}", label=fr'$\varphi={phi:.1f}$')
        ax.plot(np.real(tau_vals), z, c=f"C{j}", label=fr'$\varphi_\text{{exp}}={phi:.1f}$')
        ax.plot(np.imag(tau_vals), z, '--', c=f"C{j}")

xmin, xmax = ax.get_xlim()
for j, phi in enumerate(phis_to_plot):
        line = np.linspace(xmin, xmin*0.9, 10)
        k_val = 1/(hEk*phi)
        ax.scatter(xmin, 1/k_val, c=f"C{j}")

ax.scatter([], [], c="black", label=r'$1/k$')
ax.fill_between([xmin, xmax], 0, hEk, color="gray",  label=r"$h_\text{Ek0}$", alpha=0.15, ec="gray")
            
ax.set_xlabel(r"Momentum flux, $\tau$ [m$^2$/s$^2$]", fontsize=11)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylim(0, zmax)
ax.plot([], [], 'k-',  label=r'$\tau_x$')
ax.plot([], [], 'k--', label=r'$\tau_y$')
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)

# --- Potential well below ---
ax_pot = axes[1]
for j, phi in enumerate(phis_to_plot):  
        k_val = 1/(hEk*phi)      
        nu_z = nu0 * np.exp(-k_val * z)   # nu(z) = nu0 * exp(-kz)
        ax_pot.plot(f / nu_z, z, c=f"C{j}")
        
ax_pot.set_xlabel(r"Inverse decay scale, $\kappa$ [m$^{-2}$]", fontsize=11)
ax_pot.set_ylim(0, zmax)
ax_pot.set_xlim(0, f*100)
ax_pot.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax_pot.grid(True, linestyle='--', alpha=0.6)
axes[0].set_ylabel(r"Height, $z$ [m]", fontsize=11)
axes[1].set_ylabel(r"Height, $z$ [m]", fontsize=11)
 
plt.suptitle(r"Momentum Flux and Potential for Exponential Model",
             fontsize=14)
plt.tight_layout()
save_name="exponential_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()

#%%

from scipy.integrate import cumulative_trapezoid


nu0 = 0.1
hEk = np.sqrt(2 * nu0 / f)
phis_to_plot = [0.4, 1.0, 4.0]
Nz = 10000
zmax = 5 * hEk
z = np.linspace(0, zmax, Nz)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# Classical single-layer Ekman reference
U_theory = U0 * (1 - np.exp(-(1 + 1j) * z / hEk))
ang_theory = np.angle(U_theory, deg=True)
ax.plot(ang_theory[1:], z[1:], 'k--', lw=2, label='classical solution')


for j, phi in enumerate(phis_to_plot):
    k_val = 1 / (hEk * phi)
    print(f"k = {k_val:.2e}")
    tau_vals = tau(z, phi, k=k_val)
    nu_z = nu0 * np.exp(-k_val * z)
    U = cumulative_trapezoid(tau_vals/nu_z, z, initial=0)
    ang = np.angle(U, deg=True)   
    ax.plot(ang[1:], z[1:], c=f"C{j}", label=fr'$\varphi_\text{{exp}}={phi:.1f}$')
    
    integrand = tau_vals / nu_z
    

# --- Decorations: 1/k markers and hEk0 shading ---
xmin, xmax = ax.get_xlim()
for j, phi in enumerate(phis_to_plot):
    k_val = 1 / (hEk * phi)
    scale_height = 1 / k_val
    ax.scatter(xmin,  scale_height, c=f"C{j}", linewidth=1)

ax.fill_between([xmin, xmax], 0, hEk, color='gray', alpha=0.2, ec="black")

# Legend proxies
ax.scatter([], [], c="black", linewidth=1, label=r'$1/k$')
ax.fill_between([], [], [], color='gray', alpha=0.2, ec="black", label=r"$h_\text{Ek0}$")

ax.set_xlabel(r"Wind turning angle [°]", fontsize=11)
ax.set_ylabel(r"Height, $z$ [m]", fontsize=11)
ax.set_ylim(0, zmax)
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)

plt.suptitle(r"Spiral for Exponential Model", fontsize=14)

save_name = "exponential_angle_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()

