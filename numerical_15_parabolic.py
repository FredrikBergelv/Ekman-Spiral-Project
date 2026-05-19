import numpy as np
from scipy.integrate import solve_bvp

# -------------------------
# PARAMETERS
# -------------------------
u0 = 10.0
f = 1e-4


# -------------------------
# ν(z)
# -------------------------
def nu(z, H, nu_max):
    return nu_max * z * (H - z)

def nu_p(z, H, nu_max):
    return nu_max * (H - 2*z)

# -------------------------
# BVP SYSTEM
# -------------------------
def solve_profile(H, nu_max):

    z = np.linspace(0, H, 300)

    def fun(z, Y):
        u, up, v, vp = Y

        nu_z = nu(z, H, nu_max)
        nu_z = np.maximum(nu_z, 1e-10)

        nu_zp = nu_p(z, H, nu_max)

        return np.vstack([
            up,
            (-f*v - nu_zp*up) / nu_z,
            vp,
            ( f*(u - u0) - nu_zp*vp) / nu_z
        ])

    def bc(Y0, YH):
        return np.array([
            Y0[0],   # u_r(0)=0
            Y0[2],   # u_i(0)=0
            YH[1],   # u_r'(H)=0
            YH[3]    # u_i'(H)=0
        ])

    # initial guess (smooth decay)
    k = np.sqrt(f / (nu_max * H**2 + 1e-6))
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

import matplotlib.pyplot as plt

H_list = [1, 10, 100, 1000]
nu_values = np.linspace(1e-6, 0.5, 50)

nu_values = np.logspace(-6, 1, 50)

plt.figure()

for i, H in enumerate(H_list):
    angles = []

    for j, nu_max in enumerate(nu_values):
        z, Y = solve_profile(H, nu_max)
        surf_angle = surface_angle(z, Y)
        angles.append(surf_angle)

        percent = 100*(j + i*len(nu_values)) / (len(H_list)*len(nu_values))
        print(f"{percent:.2f}% (angle = {surf_angle:.2f} deg)")

    plt.plot(nu_values, angles, label=f"H={H}")

plt.hlines(45, min(nu_values), max(nu_values), color="black", linestyle='--', label="45° reference")

plt.xlabel(r"Maximal viscosity, $\nu_\text{max}$ [m²/s]")
plt.ylabel(r"Surface angle, $\theta$ [deg]")
plt.suptitle("Surface Angle for 1.5 parabolic model", fontsize=14)
plt.title("Surface angle vs maximal viscosity")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.xscale("log")
plt.savefig("numerical_15_parabolic.png", dpi=400)
plt.show()
