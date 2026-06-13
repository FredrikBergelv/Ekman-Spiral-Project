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

phi_values = np.logspace(-2, np.log10(4), 200)

min_viscosities  = [1e-1, 1e-2, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18]


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
def solve_profile(phi, scheme, eps):
    z = np.linspace(0, 1, 300)

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
    nu0 = nu_linear_increasing(0.5, eps)
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

schemes = ["linear increasing", "linear decreasing", "parabolic"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, scheme in zip(axes, schemes):

    print(scheme)
    for i, eps in enumerate(min_viscosities):
        print("    epsilon = ", eps)
        
        angles = []

        for phi in phi_values:
            z, Y = solve_profile(phi, scheme, eps)
            angles.append(surface_angle(z, Y))

        ax.plot(phi_values, angles, label=fr"$\epsilon={eps:.0e}$")

    ax.axhline(45, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.4)

    ax.set_title(scheme.capitalize())
    ax.set_xlabel(r"$\varphi$")

axes[0].set_ylabel(r"Surface angle $\theta$ [deg]")
axes[-1].legend()

plt.tight_layout()
save_name="numerical_15_angle_limit"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.show()