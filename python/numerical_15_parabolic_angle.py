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

# -------------------------
# ν(z)
# -------------------------
def nu(z):
    return 4*z * (1 - z)

def nu_p(z):
    return 4*(1 - 2*z)

# -------------------------
# BVP SYSTEM
# -------------------------
def solve_profile(phi):

    z = np.linspace(0, 1, 300)

    def fun(z, Y):
        u, up, v, vp = Y

        nu_z = nu(z)
        nu_z = np.maximum(nu_z, 1e-10)

        nu_zp = nu_p(z)

        return np.vstack([
            up,
            (-phi**2*v - nu_zp*up) / nu_z,
            vp,
            ( phi**2*(u - u0) - nu_zp*vp) / nu_z
        ])

    def bc(Y0, YH):
        return np.array([
            Y0[0],   # u_r(0)=0
            Y0[2],   # u_i(0)=0
            YH[1],   # u_r'(H)=0
            YH[3]    # u_i'(H)=0
        ])

    # initial guess (smooth decay)
    k = np.sqrt(phi**2 + 1e-6)
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


phi_values = np.logspace(-6, np.log10(3), 1000)


surface_angles = []
transport_angles = []
for i, phi in enumerate(phi_values):

        z, Y = solve_profile(phi)
        surf = surface_angle(z, Y)
        surface_angles.append(surf)
        
        transport= transport_angle(z, Y)
        transport_angles.append(transport)

        percent = 100* i / len(phi_values)
        print(f"{percent:.2f}% (surf = {surf:.2f} deg) (trans = {transport:.2f} deg)")

#%%
plt.figure(figsize=(8,5))
plt.plot(phi_values, surface_angles, c="C2")

plt.hlines(45, min(phi_values), max(phi_values), color="black", linestyle='--', label="45° reference")

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
plt.suptitle("Surface Angle for 1.5 Parabolic Model", fontsize=14)
plt.title("Surface angle vs layer thickness",fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(0,95)
plt.yticks([0,15,30,45,60,75,90])
save_name="numerical_15_parabolic_angle"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

#%%
plt.figure(figsize=(8,5))
plt.plot(phi_values, transport_angles, c="C2")

plt.hlines(90, min(phi_values), max(phi_values), color="black", linestyle='--', label="90° reference")

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]", fontsize=11)
plt.ylabel(r"Transport angle, $\theta_T$ [deg]", fontsize=11)
plt.suptitle("Ekman Transport Angle for 1.5 Parabolic Model", fontsize=14)
plt.title("Transport angle vs layer thickness", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(0,185)
plt.yticks([0,30,60,90,120,150,180])
save_name="numerical_15_parabolic_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

