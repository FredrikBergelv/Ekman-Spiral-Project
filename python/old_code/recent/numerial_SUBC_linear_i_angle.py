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
def solve_profile(phi, eps, prev_sol):
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
    if prev_sol is not None:
        z0 = prev_sol.x
        Y0 = prev_sol.y
    else:
        z0 = np.linspace(0, 1, 100)
        nu_mid = nu_increasing(0.5, eps)
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
prev_sol = None

for i, phi in enumerate(phi_values):
    
    surface_angles_i_now = []
    transport_angles_i_now = []
    
    for j, epsilon in enumerate(min_viscosities):
        z, Y, sol = solve_profile(phi, epsilon, prev_sol)
        
        surf = surface_angle(z, Y)
        surface_angles_i_now.append(surf)
        
        transport= transport_angle(z, Y)
        transport_angles_i_now.append(transport)
        
        prev_sol = sol
        
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

    plt.plot(phi_values, surface_angles_i[:, j], label=fr"$\epsilon={epsilon:.0e}$")
    
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
    angles = transport_angles_i[:, j]
    
    for i, ang in enumerate(angles):
        if ang<0:
            angles[i] =np.nan
        
    plt.plot(phi_values, angles, label=fr"$\epsilon={epsilon:.0e}$")

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

epsilons_to_plot = [1e-1, 1e-6]
phis_to_plot = [0.2, 1.0, 2.0]
Nz = 300
z_plot = np.linspace(0, 1, Nz)

fig, axes = plt.subplots(2, 2, figsize=(12, 6),
                         gridspec_kw={'height_ratios': [3, 1]},
                         sharey='row')

for i, epsilon in enumerate(epsilons_to_plot[:2]):  # two columns
    ax = axes[0, i]

    for j, phi in enumerate(phis_to_plot):
        
        z_sol, Y_sol, sol = solve_profile(phi, epsilon, None)

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
    ax.set_title(fr"$\epsilon = {epsilon:.0e}$", fontsize=12)
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
    ax_pot.set_xlabel(r"Inverse decay scale, $\kappa$ [m$^{-2}$]", fontsize=11)
    ax_pot.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
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

"""
#%%

u0 = 10.0
f = 1e-4

H = 1000

# -------------------------
# VISCOCITY SCHEMES
# -------------------------

def nu_increasing(z, eps):
    return np.where(z > H, eps, z/H + eps)

# -------------------------
# SOLVER
# -------------------------
def solve_profile(phi, eps, prev_sol):
    z = np.linspace(0, 5*H, 10000)

    def fun(z, Y):
        taux, taupx, tauy, taupy = Y
        nu = nu_increasing(z, eps)
        return np.vstack([
            taupx,
            -2*phi**2*tauy / (H**2*nu),
            taupy,
            2*phi**2*taux / (H**2*nu)
        ])

    def bc(Y0, Y1):
        return np.array([
            Y1[0],      # taux(0) = 0
            Y1[2],      # tauy(0) = 0
            Y0[1],      # taux'(0) = 0
            Y0[3] + f*u0/H,  # tauy'(0) = -f*u0/H
        ])

    # Classical Ekman solution as initial guess
    if prev_sol is not None:
        z0 = prev_sol.x
        Y0 = prev_sol.y
    else:
        z0 = np.linspace(0, 5*H, 10000)
        nu_mid = nu_increasing(H/2, eps)
        h_Ek = np.sqrt(2 * nu_mid / f)
        exp_decay = np.exp(-z0 / h_Ek)
        Fx_g  =  (nu_mid * u0 / h_Ek) * exp_decay * np.cos(z0 / h_Ek)
        Fy_g  = -(nu_mid * u0 / h_Ek) * exp_decay * np.sin(z0 / h_Ek)
        Fpx_g =  (nu_mid * u0 / h_Ek**2) * exp_decay * (-np.cos(z0/h_Ek) - np.sin(z0/h_Ek))
        Fpy_g =  (nu_mid * u0 / h_Ek**2) * exp_decay * (-np.sin(z0/h_Ek) + np.cos(z0/h_Ek))
        Y0 = np.vstack([Fx_g, Fpx_g, Fy_g, Fpy_g])

    sol = solve_bvp(fun, bc, z0, Y0, tol=1e-6, max_nodes=100000)
    return sol.x, sol.y, sol



from scipy.integrate import cumulative_trapezoid


epsilons_to_plot = [1e-1, 1e-6]
phis_to_plot = [0.2, 1.0, 2.0]
Nz = 10000
z_plot = np.linspace(0, 5*H, Nz)



fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

for i, epsilon in enumerate(epsilons_to_plot):
    a = ax[i]

    # Classical single-layer reference with appropriate hEk
    nu_mid = nu_increasing(H/2, epsilon)
    hEk_ref = np.sqrt(2 * nu_mid / f)
    U_theory = u0 * (1 - np.exp(-(1 + 1j) * ( z_plot) / hEk_ref))
    ang_theory = np.angle(U_theory, deg=True)
    a.plot(ang_theory[1:], z_plot[1:], 'k--', lw=2, label='classical solution')
    
    for j, phi in enumerate(phis_to_plot):
        z_sol, Y_sol, sol = solve_profile(phi, epsilon, None)
        
        taux_interp = np.interp(z_plot, z_sol, Y_sol[0])
        tauy_interp = np.interp(z_plot, z_sol, Y_sol[2])
        tau_complex = taux_interp + 1j * tauy_interp
        nu_z = nu_increasing(z_plot, epsilon)

        integrand = tau_complex / nu_z
        U = cumulative_trapezoid(integrand, z_plot, initial=0)
        
        print("U(0) = ", U[0])
        #a.plot(np.real(U), z_plot, c=f"C{j}", label=fr'$U={phi:.1f}$')
        #a.plot(np.imag(U), z_plot, c=f"C{j}",linestyle="--", label=fr'$V={phi:.1f}$')
        
        ang = np.angle(U, deg=True)
        a.plot(ang[1:], z_plot[1:], c=f"C{j}", label=fr'$\varphi={phi:.1f}$')
        
    # Decorations
    xmin, xmax = a.get_xlim()
    for j, phi in enumerate(phis_to_plot):
        a.fill_between([xmin+j*5, xmax-j*5], phi * H, color=f"C{j}", alpha=0.1)
    
    a.scatter(xmin, H, c="black")
    
    a.scatter([], [], c="black", label=r'$ H$')
    a.fill_between([], [], [], color='gray', alpha=0.2, ec="black", label=r"$h_\text{Ek}$")
    
    #a.set_ylim(0,1.1*H)
    a.set_xlabel(r"Wind turning angle [°]", fontsize=11)
    a.set_title(fr"$\epsilon = {epsilon:.0e}$", fontsize=12)
    a.grid(True, linestyle='--', alpha=0.6)
    if i == 1:
        a.legend(loc="upper right", fontsize=11)

ax[0].set_ylabel(r"Height, $z$ [m]", fontsize=11)
plt.suptitle(r"Spiral for the SUBC Linear Increasing Model", fontsize=14)
plt.tight_layout()

save_name = "numerical_SUBC_linear_i_angle_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()
"""