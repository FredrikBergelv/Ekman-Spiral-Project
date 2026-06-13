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

def nu_p_deacreasing(z):
    return -1


# -------------------------
# BVP SYSTEM
# -------------------------
def solve_profile(phi):

    z = np.linspace(0, 1, 300)

    def fun(z, Y):
        u, up, v, vp = Y

        nu_z = nu_deacreasing(z)
        nu_zp = nu_p_deacreasing(z)
    
        return np.vstack([
            up,
            (-2*phi**2*v - nu_zp*up) / nu_z,
            vp,
            ( 2*phi**2*(u - u0) - nu_zp*vp) / nu_z
            ])

    def bc(Y0, YH):
        return np.array([
            Y0[0],   # u_r(0)=0
            Y0[2],   # u_i(0)=0
            YH[1],   # u_r'(H)=0
            YH[3]    # u_i'(H)=0
            ])

    # initial guess (smooth decay)
    k = np.sqrt(phi**2 )
    u_hat = u0 * (1 - np.exp(-k*z))
    v_hat = np.zeros_like(z)

    Y0 = np.vstack([
        u_hat,
        np.gradient(u_hat, z),
        v_hat,
        np.gradient(v_hat, z)])

    sol = solve_bvp(fun, bc, z, Y0)

    return sol.x, sol.y

def surface_angle(z, Y):
    u, up, v, vp = Y

    idx = 1 # surface 

    angle = np.arctan2(vp[idx], up[idx])
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


