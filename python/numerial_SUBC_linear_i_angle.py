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
        Fx, Fpx, Fy, Fpy = Y
        nu = nu_increasing(z, eps)
        return np.vstack([
            Fpx,
            -2*phi**2*Fy / nu,
            Fpy,
            2*phi**2*Fx / nu
        ])

    def bc(Y0, Y1):
        return np.array([
            Y1[0],      # Fx(0) = 0
            Y1[2],      # Fy(0) = 0
            Y0[1],      # Fx'(0) = 0
            Y0[3] + f*u0,  # Fy'(0) = -f*u0
        ])

    # Classical Ekman solution as initial guess
    nu0 = nu_increasing(0.5, eps)
    h_Ek = np.sqrt(2 * nu0 / f)
    Fx_guess = (nu0 * u0 / h_Ek) * np.exp(-z / h_Ek) * np.cos(z / h_Ek)
    Fy_guess = -(nu0 * u0 / h_Ek) * np.exp(-z / h_Ek) * np.sin(z / h_Ek)
    Fpx_guess = (nu0 * u0 / h_Ek**2) * np.exp(-z / h_Ek) * (-np.cos(z / h_Ek) - np.sin(z / h_Ek))
    Fpy_guess = (nu0 * u0 / h_Ek**2) * np.exp(-z / h_Ek) * (-np.sin(z / h_Ek) + np.cos(z / h_Ek))
    
    Y_init = np.vstack([
        Fx_guess,
        Fpx_guess,
        Fy_guess,
        Fpy_guess
    ])


    sol = solve_bvp(fun, bc, z, Y_init)
    return sol.x, sol.y

# -------------------------
# ANGLE METRIC
# -------------------------
def surface_angle(z, Y):
    Fx, Fpx, Fy, Fpy = Y
    idx = 1  # surface (top boundary)

    angle = np.arctan2(Fy[idx], Fx[idx])
    return np.degrees(angle)

# -------------------------
# PLOTTING
# -------------------------
def transport_angle(z, Y):
    Fx, Fpx, Fy, Fpy = Y

    # T = i/f F(z=0) gives
    Fx0 = Fx[0]
    Fy0 = Fy[0]
    angle = np.arctan2(Fx0, -Fy0)

    return np.degrees(angle)

extent = 4
phi_values = np.logspace(-6, np.log10(extent), 1000)
phi_values = np.linspace(1e-6, extent, 200)


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

    plt.plot(phi_values, surface_angles_i[:, j], label=fr"$\epsilon={epsilon:.0e}$")
    
plt.hlines(45, min(phi_values), max(phi_values), color="black", linestyle='--', label="45° reference")

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
plt.suptitle("Surface Angle for the the SUBC Linear Increasing Model", fontsize=14)
plt.title("Surface angle vs layer thickness", fontsize=13)
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
    plt.plot(phi_values, transport_angles_i[:, j],  label=fr"$\epsilon={epsilon:.0e}$")

plt.hlines(135, min(phi_values), max(phi_values), color="black", linestyle='--', label="135° reference")

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]", fontsize=11)
plt.ylabel(r"Transport angle, $\theta_T$ [deg]", fontsize=11)
plt.suptitle("Ekman Transport Angle the SUBC Linear Increasing Model", fontsize=14)
plt.title("Transport angle vs layer thickness", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(90,185)
plt.yticks([90, 105, 120, 135, 150, 165, 180])
plt.xticks(np.arange(0, extent + 0.5, 0.5))

save_name="numerical_SUBC_linear_i_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

