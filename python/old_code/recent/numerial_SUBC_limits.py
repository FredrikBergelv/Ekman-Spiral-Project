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
u0 = 100.0
f = 1e-4

extent = 250
phi_values = np.linspace(0, extent, 200)


min_viscosities  = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
min_viscosities  = [1e-3, 1e-6]




# -------------------------
# VISCOCITY SCHEMES
# -------------------------

def nu_linear_increasing(z, eps):
    return z + eps

def nu_linear_decreasing(z, eps):
    return (1 - z) + eps

def nu_parabolic(z, eps):
    return 4*z*(1 - z) + eps

# -------------------------
# SOLVER
# -------------------------
def solve_profile(phi, scheme, eps, prev_sol=None):

    def fun(z, Y):
        Fx, Fpx, Fy, Fpy = Y
        
        if scheme == "linear increasing":
            nu = nu_linear_increasing(z, eps)

        elif scheme == "linear decreasing":
            nu = nu_linear_decreasing(z, eps)

        elif scheme == "parabolic":
            nu = nu_parabolic(z, eps)

        else:
            raise ValueError("Unknown scheme")
            
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
    if prev_sol is not None:
        z0 = prev_sol.x
        Y0 = prev_sol.y
    else:
        z0 = np.linspace(0, 1, 100)
        nu_mid = nu_linear_increasing(0.5, eps)
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
    Fx, Fpx, Fy, Fpy = Y
    idx = 1  # surface (top boundary)

    angle = np.arctan2(Fy[idx], Fx[idx])
    return np.degrees(angle)

# -------------------------
# PLOTTING
# -------------------------

schemes = ["linear increasing", "parabolic"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True, sharex=True)

prev_sol = None
for ax, scheme in zip(axes, schemes):
    
    print(scheme)

    for i, eps in enumerate(min_viscosities):
        print("    epsilon = ", eps)
        
        angles = []

        for phi in phi_values:
            z, Y, sol = solve_profile(phi, scheme, eps, prev_sol)
            
            val = surface_angle(z, Y)
            
            prev_sol = sol

            angles.append(val)
            
        if i == 0:
            ax.hlines(45, min(phi_values), max(phi_values), color="black", linestyle='--', label="45° reference")
            
        ax.plot(phi_values, angles, label=fr"$\epsilon={eps:.0e}$ ms$^{{-1}}$")
        
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.4)

    ax.set_title(scheme.capitalize())
    ax.set_xlabel(r"Dimensionless layer thickness, $\varphi$ [-]",fontsize=11)

axes[0].set_ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
axes[-1].legend(fontsize=11, loc="upper right")

plt.suptitle("Surface Angle for SUBC Increasing Models", fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[1].grid(True, linestyle='--', alpha=0.6)



save_name="numerical_SUBC_angle_limit"
#plt.savefig(f"plots/{save_name}.png", dpi=400)
#plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()