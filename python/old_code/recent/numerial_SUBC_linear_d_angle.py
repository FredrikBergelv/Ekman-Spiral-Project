"""
Created on Mon May  4 07:53:05 2026

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


# -------------------------
# ν(z)
# -------------------------
min_viscosity = 1e-13
def nu_decreasing(z, epsilon=min_viscosity):
    return (1 - z) + epsilon

# -------------------------
# SOLVER
# -------------------------
def solve_profile(phi, epsilon=min_viscosity):
    z = np.linspace(0, 1, 300)

    def fun(z, Y):
        taux, taupx, tauy, taupy = Y
        nu = nu_decreasing(z, epsilon)
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
    z0 = np.linspace(0, 1, 100)
    nu_mid = nu_decreasing(0.5, min_viscosity)
    h_Ek = np.sqrt(2 * nu_mid / f)
    exp_decay = np.exp(-z0 / h_Ek)
    Fx_g  =  (nu_mid * u0 / h_Ek) * exp_decay * np.cos(z0 / h_Ek)
    Fy_g  = -(nu_mid * u0 / h_Ek) * exp_decay * np.sin(z0 / h_Ek)
    Fpx_g =  (nu_mid * u0 / h_Ek**2) * exp_decay * (-np.cos(z0/h_Ek) - np.sin(z0/h_Ek))
    Fpy_g =  (nu_mid * u0 / h_Ek**2) * exp_decay * (-np.sin(z0/h_Ek) + np.cos(z0/h_Ek))
    Y0 = np.vstack([Fx_g, Fpx_g, Fy_g, Fpy_g])

    sol = solve_bvp(fun, bc, z0, Y0, tol=1e-8, max_nodes=10000)
    return sol.x, sol.y, sol

# -------------------------
# ANGLE METRIC
# -------------------------
def surface_angle(z, Y):
    taux, taupx, tauy, taupy = Y
    idx = 1  # surface (top boundary)

    angle = np.arctan2(tauy[idx], taux[idx])
    return np.degrees(angle)

def transport_angle(z, Y):
    taux, taupx, tauy, taupy = Y

    # T = i/f tau(z=0) gives
    taux0 = taux[0]
    tauy0 = tauy[0]
    angle = np.arctan2(taux0, -tauy0)

    return np.degrees(angle)

extent=4
phi_values = np.logspace(-2, np.log10(extent), 1000)
phi_values = np.linspace(0, extent, 200)



surface_angles_d = []
transport_angles_d = []


for i, phi in enumerate(phi_values):

        
        z_d, Y_d,sol = solve_profile(phi)
        surf_angle_d = surface_angle(z_d, Y_d)
        surface_angles_d.append(surf_angle_d)
        
        trans_angle_d = transport_angle(z_d, Y_d)
        transport_angles_d.append(trans_angle_d)
        
        #if surf_angle_i<45.0001:
            #print("Bingo!, varphi = ", phi)
            #break
        
        percent = 100* i / len(phi_values)
        print(f"{percent:.2f}% (surf = {surf_angle_d:.2f} deg) ")

#%%
plt.figure(figsize=(8,5))
plt.plot(phi_values, surface_angles_d)

plt.hlines(45, min(phi_values), max(phi_values), color="black", linestyle='--', label="45° reference")

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
plt.suptitle("Surface Angle for SUBC Linear Decreasing Model", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(0,95)
plt.yticks([0, 15, 30, 45, 60, 75, 90])
plt.xticks(np.arange(0, extent + 0.5, 0.5))
save_name="numerical_SUBC_linear_d_angle"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

#%%
plt.figure(figsize=(8,5))
plt.plot(phi_values[1:], transport_angles_d[1:])

plt.hlines(135, min(phi_values), max(phi_values), color="black", linestyle='--', label="135° reference")

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]", fontsize=11)
plt.ylabel(r"Transport angle, $\theta_T$ [deg]", fontsize=11)
plt.suptitle("Ekman Transport Angle for the SUBC Linear Decreasing Model", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(90,185)
plt.yticks([90, 105, 120, 135, 150, 165, 180])
plt.xticks(np.arange(0, extent + 0.5, 0.5))

save_name="numerical_SUBC_linear_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

#%%
nu0 = 1
epsilons_to_plot = 1e-3
phis_to_plot = [0.2, 1.0, 2.0]
Nz = 300
z_plot = np.linspace(0, 1, Nz)

fig, axes = plt.subplots(2, 1, figsize=(6, 6),
                         gridspec_kw={'height_ratios': [3, 1]},
                         sharey='row')

ax = axes[0]

for j, phi in enumerate(phis_to_plot):
        z_sol, Y_sol, sol = solve_profile(phi, epsilons_to_plot)

        taux = Y_sol[0]
        tauy = Y_sol[2]

        # Interpolate onto uniform z_plot grid
        taux_interp = np.interp(z_plot, z_sol, taux)
        tauy_interp = np.interp(z_plot, z_sol, tauy)

        ax.plot(taux_interp, z_plot, c=f"C{j}", label=fr'$\varphi={phi:.1f}$')
        ax.plot(tauy_interp, z_plot, '--', c=f"C{j}")
        
        minval, maxval = ax.get_xlim()
        #minval = min(min(taux_interp), min(tauy_interp))
        #maxval= max(max(taux_interp), max(tauy_interp))
        ax.fill_between([minval, maxval], 0, 1/phi, color=f"C{j}", alpha=0.15)

ax.fill_between([], [], color='gray', alpha=0.15, label=r'$h_\text{Ek}$')

ax.set_xlabel(r"Momentum flux, $\tau$ [m$^2$/s$^2$]", fontsize=11)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylim(0, 1.1)
ax.plot([], [], 'k-',  label=r'$\tau_x$')
ax.plot([], [], 'k--', label=r'$\tau_y$')
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)

    # --- Potential well below ---
ax_pot = axes[1]
    
nu_z = nu_decreasing(z_plot, epsilons_to_plot)
  
ax_pot.plot(f / nu_z, z_plot,  'k')
ax_pot.set_xlabel(r"Inverse decay scale, $\kappa$ [m$^{-2}$]", fontsize=11)
ax_pot.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax_pot.set_ylim(0, 1)
ax_pot.set_xlim(0, f*10)
ax_pot.grid(True, linestyle='--', alpha=0.6)

axes[0].set_ylabel(r"Norm. height, $\tilde z$ [-]", fontsize=11)
axes[1].set_ylabel(r"Norm. height, $\tilde z$ [-]", fontsize=11)

# Same xlim per row


plt.suptitle(r"Momentum Flux and Potential for SUBC Linear Decreasing Model",
             fontsize=13)
plt.tight_layout()
save_name = "numerical_SUBC_linear_d_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()




