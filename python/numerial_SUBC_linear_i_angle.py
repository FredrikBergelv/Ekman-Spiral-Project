"""
Created on Sat May  2 23:16:29 2026

@author: fredrik
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------
u0 = 10.0
f = 1e-4

min_viscosities = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15]

# -------------------------
# VISCOCITY SCHEMES
# -------------------------

def nu_increasing(z, eps):
    return z + eps

# -------------------------
# SOLVER
# -------------------------
def solve_profile(phi, eps):
    z = np.linspace(0, 1, 300)

    def fun(z, Y):
        taux, taupx, tauy, taupy = Y
        nu = nu_increasing(z, eps)
        return np.vstack([
            taupx,
            -2*phi**2*tauy / nu,
            taupy,
            2*phi**2*taux / nu
        ])

    def bc(Y0, Y1):
        return np.array([
            Y1[0],      # taux(0) = 0
            Y1[2],      # tauy(0) = 0
            Y0[1],      # taux'(0) = 0
            Y0[3] + f*u0,  # tauy'(0) = -f*u0
        ])

    # Classical Ekman solution as initial guess
    nu0 = nu_increasing(0.5, eps)
    h_Ek = np.sqrt(2 * nu0 / f)
    taux_guess = (nu0 * u0 / h_Ek) * np.exp(-z / h_Ek) * np.cos(z / h_Ek)
    tauy_guess = -(nu0 * u0 / h_Ek) * np.exp(-z / h_Ek) * np.sin(z / h_Ek)
    taupx_guess = (nu0 * u0 / h_Ek**2) * np.exp(-z / h_Ek) * (-np.cos(z / h_Ek) - np.sin(z / h_Ek))
    taupy_guess = (nu0 * u0 / h_Ek**2) * np.exp(-z / h_Ek) * (-np.sin(z / h_Ek) + np.cos(z / h_Ek))
    
    Y_init = np.vstack([
        taux_guess,
        taupx_guess,
        tauy_guess,
        taupy_guess
    ])


    sol = solve_bvp(fun, bc, z, Y_init)
    return sol.x, sol.y

# -------------------------
# ANGLE METRIC
# -------------------------
def surface_angle(z, Y):
    taux, taupx, tauy, taupy = Y
    idx = 1  # surface (top boundary)

    angle = np.arctan2(tauy[idx], taux[idx])
    return np.degrees(angle)

# -------------------------
# PLOTTING
# -------------------------
def transport_angle(z, Y):
    taux, taupx, tauy, taupy = Y

    # T = i/f tau(z=0) gives
    taux0 = taux[0]
    tauy0 = tauy[0]
    angle = np.arctan2(taux0, -tauy0)

    return np.degrees(angle)

extent = 4
phi_values = np.linspace(0, extent, 200)


surface_angles_i = []
transport_angles_i = []
for i, phi in enumerate(phi_values):
    
    surface_angles_i_now = []
    transport_angles_i_now = []
    
    for j, epsilon in enumerate(min_viscosities):
        z, Y = solve_profile(phi, epsilon)
        
        surf = surface_angle(z, Y)
        surface_angles_i_now.append(surf)
        
        transport= transport_angle(z, Y)
        transport_angles_i_now.append(transport)
        
        #if surf<45.0001:
            #print("Bingo!, varphi = ", phi)
            #break

        percent = 100 * (i * len(min_viscosities) + j + 1) / (len(phi_values) * len(min_viscosities))
        print(f"{percent:.2f}% (surf = {surf:.2f} deg)")
        
    surface_angles_i.append([surface_angles_i_now])
    transport_angles_i.append(transport_angles_i_now)
    
        
#%%
plt.figure(figsize=(8,5))

surface_angles_i = np.squeeze(np.array(surface_angles_i))
for j, epsilon in enumerate(min_viscosities):

    plt.plot(phi_values, surface_angles_i[:, j], label=fr"$\epsilon={epsilon:.0e}$ ms$^{{-1}}$")
    
plt.hlines(45, min(phi_values), max(phi_values), color="black", linestyle='--', label="45° reference")

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
plt.suptitle("Surface Angle for the the SUBC Linear Increasing Model", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(0,95)
plt.yticks([0, 15, 30, 45, 60, 75, 90])
plt.xticks(np.arange(0, extent + 0.5, 0.5))
save_name="numerical_SUBC_linear_i_angle"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

#%%
plt.figure(figsize=(8,5))
transport_angles_i = np.squeeze(np.array(transport_angles_i))
for j, epsilon in enumerate(min_viscosities):
    plt.plot(phi_values, transport_angles_i[:, j], label=fr"$\epsilon={epsilon:.0e}$ ms$^{{-1}}$")

plt.hlines(135, min(phi_values), max(phi_values), color="black", linestyle='--', label="135° reference")

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]", fontsize=11)
plt.ylabel(r"Transport angle, $\theta_T$ [deg]", fontsize=11)
plt.suptitle("Ekman Transport Angle the SUBC Linear Increasing Model", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(90,185)
plt.yticks([90, 105, 120, 135, 150, 165, 180])
plt.xticks(np.arange(0, extent + 0.5, 0.5))

save_name="numerical_SUBC_linear_i_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

#%%

epsilons_to_plot = [1e-1, 1e-3]
phis_to_plot = [0.2, 1.0, 2.0]
Nz = 300
z_plot = np.linspace(0, 1, Nz)

fig, axes = plt.subplots(2, 2, figsize=(12, 6),
                         gridspec_kw={'height_ratios': [3, 1]},
                         sharey='row')

for i, epsilon in enumerate(epsilons_to_plot[:2]):  # two columns
    ax = axes[0, i]

    for j, phi in enumerate(phis_to_plot):
        
        z_sol, Y_sol = solve_profile(phi, epsilon)

        taux = Y_sol[0]
        tauy = Y_sol[2]

        # Interpolate onto uniform z_plot grid
        taux_interp = np.interp(z_plot, z_sol, taux)
        tauy_interp = np.interp(z_plot, z_sol, tauy)

        ax.plot(taux_interp, z_plot, c=f"C{j}", label=fr'$\varphi={phi:.1f}$')
        ax.plot(tauy_interp, z_plot, '--', c=f"C{j}")
        minval, maxval = ax.get_xlim()
        ax.fill_between([minval, maxval], 0, 1/phi, color=f"C{j}", alpha=0.15)

    ax.fill_between([], [], color='gray', alpha=0.15, label=r'$h_\text{Ek}$')

    ax.set_xlabel(r"Momentum flux, $\tau$ [m$^2$/s$^2$]", fontsize=11)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_title(fr"$\epsilon = {epsilon:.0e}$ ms$^{{-1}}$", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.plot([], [], 'k-',  label=r'$\tau_x$')
    ax.plot([], [], 'k--', label=r'$\tau_y$')
    if i==1:
        ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Potential well below ---
    ax_pot = axes[1, i]
    nu_z = nu_increasing(z_plot, epsilon)
    ax_pot.plot(1 / nu_z, z_plot, 'k')
    ax_pot.plot(f / nu_z, z_plot, 'k')
    ax_pot.set_xlabel(r"Eigenvalue, $\lambda$ [m$^{-2}$]", fontsize=11)
    ax_pot.set_ylim(0, 1)
    ax_pot.set_xlim(0, f*10)
    ax_pot.grid(True, linestyle='--', alpha=0.6)

# Shared y-labels on left column only
axes[0, 0].set_ylabel(r"Norm. height, $\tilde z$ [-]", fontsize=11)
axes[1, 0].set_ylabel(r"Norm. height, $\tilde z$ [-]", fontsize=11)

# Same xlim per row


plt.suptitle(r"Momentum Flux and Potential for SUBC Linear Increasing Model",
             fontsize=14)
plt.tight_layout()
save_name = "numerical_SUBC_linear_i_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()


