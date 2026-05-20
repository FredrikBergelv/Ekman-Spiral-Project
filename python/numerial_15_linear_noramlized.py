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
def nu_deacreasing(z):
    return (1 - z)

def nu_p_deacreasing(z):
    return -1

def nu_increasing(z):
    return z

def nu_p_increasing(z):
    return 1

# -------------------------
# BVP SYSTEM
# -------------------------
def solve_profile(phi, scheme):

    z = np.linspace(0, 1, 300)

    def fun(z, Y):
        u, up, v, vp = Y

        if scheme == "decreasing":
            nu_z = nu_deacreasing(z)
            nu_z = np.maximum(nu_z, 1e-10)
            nu_zp = nu_p_deacreasing(z)
            
        elif scheme == "increasing" :
            nu_z = nu_increasing(z)
            nu_z = np.maximum(nu_z, 1e-10)
            nu_zp = nu_p_increasing(z)
        
        else:
            print("choose right scheme!")

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


phi_values = np.logspace(-6, 0.3, 100)


plt.close("all")
plt.figure()

angles_d = []
angles_i = []

for i, phi in enumerate(phi_values):

        z_i, Y_i = solve_profile(phi, "increasing")
        surf_angle_i = surface_angle(z_i, Y_i)
        angles_i.append(surf_angle_i)
        
        z_d, Y_d = solve_profile(phi, "decreasing")
        surf_angle_d = surface_angle(z_d, Y_d)
        angles_d.append(surf_angle_d)
        
        
        percent = 100* i / len(phi_values)
        print(f"{percent:.2f}% (angle = {surf_angle_i:.2f} deg and {surf_angle_d:.2f} deg)")

plt.plot(phi_values, angles_i, label="linear increasing")
plt.plot(phi_values, angles_d, label="linear decreasing")


plt.hlines(45, min(phi_values), max(phi_values), color="black", linestyle='--', label="45° reference")

plt.xlabel(r"Lower layer thickness, $\varphi$ [-]")
plt.ylabel(r"Surface angle, $\hat\theta$ [deg]")
plt.suptitle("Surface Angle for 1.5 linear model", fontsize=14)
plt.title("Surface angle vs Lower layer thickness")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
#plt.xscale("log")
plt.ylim(0,95)
plt.savefig("numerical_15_linear_noramlized.png", dpi=400)
plt.show()
