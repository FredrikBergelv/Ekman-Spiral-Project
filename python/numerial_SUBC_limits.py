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

phi_values = np.logspace(-2, np.log10(4), 60)

min_viscosities  = [1e-12, 1e-19]


# -------------------------
# VISCOCITY SCHEMES
# -------------------------

def nu_linear_increasing(z, eps):
    return z + eps

def nup_linear_increasing(z):
    return 1.0

def nu_linear_decreasing(z, eps):
    return (1 - z) + eps

def nup_linear_decreasing(z):
    return -1.0

def nu_parabolic(z, eps):
    return 4*z*(1 - z) + eps

def nup_parabolic(z):
    return 4*(1 - 2*z)

# -------------------------
# SOLVER
# -------------------------

def solve_profile(phi, scheme, eps):
    z = np.linspace(0, 1, 300)

    def fun(z, Y):
        u, up, v, vp = Y

        if scheme == "linear increasing":
            nu = nu_linear_increasing(z, eps)
            nu_p = nup_linear_increasing(z)

        elif scheme == "linear decreasing":
            nu = nu_linear_decreasing(z, eps)
            nu_p = nup_linear_decreasing(z)

        elif scheme == "parabolic":
            nu = nu_parabolic(z, eps)
            nu_p = nup_parabolic(z)

        else:
            raise ValueError("Unknown scheme")

        return np.vstack([
            up,
            (-2*phi**2*v - nu_p*up) / nu,
            vp,
            (2*phi**2*(u - u0) - nu_p*vp) / nu
        ])

    def bc(Y0, Y1):
        return np.array([
            Y0[0],  # u(0)=0
            Y0[2],  # v(0)=0
            Y1[1],  # u'(1)=0
            Y1[3]   # v'(1)=0
        ])

    k = max(phi, 1e-6)
    u_guess = u0 * (1 - np.exp(-k*z))
    v_guess = np.zeros_like(z)

    Y_init = np.vstack([
        u_guess,
        np.gradient(u_guess, z),
        v_guess,
        np.gradient(v_guess, z)
    ])

    sol = solve_bvp(fun, bc, z, Y_init)
    return sol.x, sol.y

# -------------------------
# ANGLE METRIC
# -------------------------

def surface_angle(z, Y):
    u, up, v, vp = Y
    idx = 1  # surface (top boundary)

    angle = np.arctan2(vp[idx], up[idx])
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

        ax.plot(phi_values, angles, label=fr"$\epsilon={eps:.0e}$", marker="o")

    ax.axhline(45, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.4)

    ax.set_title(scheme.capitalize())
    ax.set_xlabel(r"$\varphi$")

axes[0].set_ylabel(r"Surface angle $\theta$ [deg]")
axes[-1].legend()

plt.tight_layout()
save_name="numerical_15_angle_limit"
#plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.show()