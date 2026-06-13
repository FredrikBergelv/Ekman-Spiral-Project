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
def nu_deacreasing(z):
    return (1 - z) + min_viscosity

# -------------------------
# SOLVER
# -------------------------
def solve_profile(phi):
    z = np.linspace(0, 1, 300)

    def fun(z, Y):
        Fx, Fpx, Fy, Fpy = Y
        nu = nu_deacreasing(z)
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
    nu0 = nu_deacreasing(0.5)
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

def transport_angle(z, Y):

    u, up, v, vp = Y

    # transport relative to geostrophic flow
    transport_u = np.trapz(u - u0, z)
    transport_v = np.trapz(v, z)

    angle = np.arctan2(transport_v, transport_u)

    return np.degrees(angle)

extent=4
phi_values = np.logspace(-2, np.log10(extent), 1000)
phi_values = np.linspace(1e-2, extent, 200)



surface_angles_d = []
transport_angles_d = []


for i, phi in enumerate(phi_values):

        
        z_d, Y_d = solve_profile(phi)
        surf_angle_d = surface_angle(z_d, Y_d)
        surface_angles_d.append(surf_angle_d)
        
        
        #trans_angle_d = transport_angle(z_d, Y_d)
        #transport_angles_d.append(trans_angle_d)
        
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
plt.title("Surface angle vs layer thickness", fontsize=13)
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
plt.plot(phi_values, transport_angles_d, label="linear decreasing")

plt.hlines(135, min(phi_values), max(phi_values), color="black", linestyle='--', label="135° reference")

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]", fontsize=11)
plt.ylabel(r"Transport angle, $\theta_T$ [deg]", fontsize=11)
plt.suptitle("Ekman Transport Angle for 1.5 Linear Model", fontsize=14)
plt.title("Transport angle vs layer thickness", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(90,185)
plt.yticks([90, 105, 120, 135, 150, 165, 180])
plt.xticks(np.arange(0, extent + 0.5, 0.5))

save_name="numerical_SUBC_linear_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()


