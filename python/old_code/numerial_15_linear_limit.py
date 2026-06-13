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
minval = 1e-18
def nu_deacreasing(z):
    return (1 - z) + minval

def nu_p_deacreasing(z):
    return -1

def nu_increasing(z):
    return z + minval

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
            #nu_z = np.maximum(nu_z,0)
            nu_zp = nu_p_deacreasing(z)
            
        elif scheme == "increasing" :
            nu_z = nu_increasing(z)
            #nu_z = np.maximum(nu_z, 1e-18)
            nu_zp = nu_p_increasing(z)
        
        else:
            print("choose right scheme!")

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
phi_values = np.logspace(-2, np.log10(extent), 100)


surface_angles_d = []
surface_angles_i = []

minvals = [1e-4, 1e-7, 1e-10, 1e-13, 1e-16, 1e-19]

plt.figure(figsize=(8,5))

for minval in minvals:

    surface_angles_i = []

    for phi in phi_values:

        # overwrite viscosity functions for this run
        def solve_profile(phi, scheme):

            z = np.linspace(0, 1, 300)

            def fun(z, Y):

                u, up, v, vp = Y

                if scheme == "increasing":
                    nu_z = z + minval
                    nu_zp = 1

                elif scheme == "decreasing":
                    nu_z = (1 - z) + minval
                    nu_zp = -1

                return np.vstack([
                    up,
                    (-2*phi**2*v - nu_zp*up) / nu_z,
                    vp,
                    (2*phi**2*(u-u0) - nu_zp*vp) / nu_z
                ])

            def bc(Y0, YH):
                return np.array([
                    Y0[0],
                    Y0[2],
                    YH[1],
                    YH[3]
                ])

            k = max(phi, 1e-6)

            u_hat = u0*(1-np.exp(-k*z))
            v_hat = np.zeros_like(z)

            Y0 = np.vstack([
                u_hat,
                np.gradient(u_hat, z),
                v_hat,
                np.gradient(v_hat, z)
            ])

            sol = solve_bvp(fun, bc, z, Y0)

            return sol.x, sol.y

        z_i, Y_i = solve_profile(phi, "increasing")
        surface_angles_i.append(surface_angle(z_i, Y_i))

    plt.plot(
        phi_values,
        surface_angles_i,
        label=fr"$\nu=z+{minval:.0e}$"
    )

plt.hlines(
    45,
    phi_values.min(),
    phi_values.max(),
    color="black",
    linestyle="--",
    label="45° reference"
)

plt.xlabel(r"Dimensionless layer thickness, $\varphi$")
plt.ylabel(r"Surface angle, $\theta$ [deg]")
plt.title("Sensitivity to lower viscosity cutoff")
plt.grid(True, alpha=0.5)
plt.legend()
plt.ylim(0,95)
save_name="numerical_15_linear_angle_limit"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.show()