#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:01:30 2025

@author: fredrik

Fixed Ekman BVP script
- vectorized, robust nu(z)
- correct coupling a(z)=f/nu(z)
- correct decay rate k = sqrt(a)/sqrt(2)
- correct Robin BCs: u' + k u = 0, v' + k v = 0 at z_max
- consistent plotting (no accidental +u0 shifts)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# -------------------------
# Parameters / grid
# -------------------------
u0      = 5.0                                # Thermal-wind scale (m/s)
Omega   = 2*np.pi / (24*3600)                # Earth's rotation rate (rad/s)
lat_rad = np.deg2rad(45.0)
f       = 2 * Omega * np.sin(lat_rad)        # Coriolis parameter at 

z_max   = 400      # truncation depth (m)
z_surf  = 1        # At which height is surface? (for wind)
nz      = 5000       # grid points
z_heights = np.linspace(0.0, z_max, nz)

# Prandtl parameters for nu(z)
k_karman = 0.4
u_friction = 0.4    # friction velocity u*
h_BL = 100.0        # boundary-layer height H (m)

background_viscosity = 0.4

# -------------------------
# nu(z): Prandtl parabolic eddy viscosity (vectorized)
# -------------------------
def nu(z):
    """
    Prandtl parabolic eddy-viscosity profile, vectorized.
    nu(z) = max(molecular_viscosity, kappa * u_* * z * (1 - z/H))
    Works for scalar or numpy array z.
    """
    z = np.asarray(z, dtype=float)
    nu_par = background_viscosity + k_karman*u_friction*z*np.exp(-z/30)

    return nu_par


# -------------------------
# coupling a(z) = f / nu(z)
# -------------------------
def a_coupling(z):
    z = np.asarray(z, dtype=float)
    return f / nu(z)


# -------------------------
# decay rate k(z) = sqrt(a)/sqrt(2)
# -------------------------
def decay_rate(z):
    a = a_coupling(z)
    return np.sqrt(a) / np.sqrt(2.0)


# -------------------------
# Convert to first-order system
# Y = [u, up, v, vp]
# Y' = [ up, -a(z)*v, vp, a(z)*u ]
# -------------------------
def fun(z, Y):
    u = Y[0]
    up = Y[1]
    v = Y[2]
    vp = Y[3]
    a = a_coupling(z)   # a can be scalar or array; broadcasting works
    return np.vstack((up,
                      -a * v,
                      vp,
                      a * u))


# -------------------------
# Boundary conditions
# -------------------------
def bc(Y0, Yf):
    # evaluate decay rate at z_max (scalar)
    k_lim = decay_rate(z_max)
    u0_val, up0_val, v0_val, vp0_val = Y0
    uf_val, upf_val, vf_val, vpf_val = Yf
    return np.array([
        v0_val - 0.0,        # v(0)=0
        u0_val + u0,         # u(0) = -u0
        upf_val + k_lim * uf_val,   # u'(zmax) + k u(zmax) = 0
        vpf_val + k_lim * vf_val    # v'(zmax) + k v(zmax) = 0
    ])


# -------------------------
# initial guess (physically motivated)
# -------------------------
# estimate mean k from mean a
k_mean = np.sqrt(np.mean(a_coupling(z_heights))) / np.sqrt(2.0)

u_guess = -u0 * np.exp(-k_mean * z_heights) * np.cos(k_mean * z_heights)
v_guess =  u0 * np.exp(-k_mean * z_heights) * np.sin(k_mean * z_heights)
up_guess = np.gradient(u_guess, z_heights)
vp_guess = np.gradient(v_guess, z_heights)
Y_guess = np.vstack((u_guess, up_guess, v_guess, vp_guess))


# -------------------------
# Solve BVP
# -------------------------
sol = solve_bvp(fun, bc, z_heights, Y_guess, max_nodes=20000)
if sol.status != 0:
    print("Warning: solve_bvp did not fully converge. status:", sol.status, "message:", sol.message)

u_sol = sol.y[0]
v_sol = sol.y[2]
z_sol = sol.x


# -------------------------
# Angle at surface
# -------------------------
idx = np.argmin(np.abs(z_sol - z_surf))
theta0 = np.arctan2(np.abs(v_sol[idx]-v_sol[0]), np.abs(u_sol[idx]-u_sol[0]))
theta_deg = np.degrees(theta0)
print(f"Angle at the surface: {theta_deg:.0f} degrees")



# -------------------------
# Prepare analytic constant-nu solution for comparison
# -------------------------
nu_inft = background_viscosity
a_const = f / nu_inft
k_const = np.sqrt(a_const) / np.sqrt(2.0)

u_old = -u0 * np.exp(-k_const * z_sol) * np.cos(k_const * z_sol)
v_old =  u0 * np.exp(-k_const * z_sol) * np.sin(k_const * z_sol)


# -------------------------
# Plotting
# -------------------------
plt.close("all")
fig, axs = plt.subplots(1,3, figsize=(12,5))
fig.suptitle(r'Prandtl eddy-viscosity: $\nu(z)=\kappa u_* ze^{z/30}$', fontsize=15)

ax = axs[0]
ax.plot(nu(z_sol), z_sol, color='green', label=r'$\nu_\text{Prandtl}(z)$')
ax.plot(nu_inft+0*z_sol, z_sol, '--', color='green', label=r'$\nu_\text{Karman}$')
ax.set_xlabel(r"$\nu(z)$ [m²/s]", fontsize=12)
ax.set_ylabel("Height [m]", fontsize=12)
ax.set_title("Kinematic Viscosity Profile", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc="upper center", fontsize=12)
ax.text(0.97, 0.95, "(a)", transform=ax.transAxes, fontsize=12, fontname='DejaVu Sans', ha='right', va='top')


ax = axs[1]
ax.plot(u0+u_sol, z_sol, label=r"$u(z, \nu_\text{Pra.})$", c="C0")
ax.plot(u0+u_old, z_sol, label=r"$u(z, \nu_\text{Kar.})$", linestyle="--", c="C0")
ax.plot(v_sol, z_sol, label=r"$v(z, \nu_\text{Pra.})$", c="C1")
ax.plot(v_old, z_sol, label=r"$v(z, \nu_\text{Kar.})$",linestyle="--", c="C1")
ax.text(0.97, 0.95, "(b)", transform=ax.transAxes, fontsize=12, fontname='DejaVu Sans', ha='right', va='top')


ax.set_xlabel("velocity [m/s]", fontsize=12)
ax.set_ylabel("Height [m]", fontsize=12)
ax.set_title("Ekman Spiral vs Height", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)

ax = axs[2]
ax.plot(u0+u_sol, v_sol, '-k', label=r"$u(\nu_\text{Pra.})$ vs $v(\nu_\text{Pra.})$")
ax.plot(u0+u_old, v_old, '-k',linestyle="--", c="black", label=r"$u(\nu_\text{Kar.})$ vs $v(\nu_\text{Kar.})$")
ax.scatter(u0+u_sol[0], v_sol[0], color='red', label='surface')
ax.scatter(u0+u_sol[-1], v_sol[-1], color='green', label='top')
ax.set_xlabel("u [m/s]", fontsize=12)
ax.set_ylabel("v [m/s]", fontsize=12)
ax.set_title("Ekman Spiral Viewed from Above", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)
ax.axis('equal')
ax.text(0.97, 0.95, "(c)", transform=ax.transAxes, fontsize=12, fontname='DejaVu Sans', ha='right', va='top')


angle_text = f"Wind angle: {theta_deg:.0f}°"
ax.text(0.05, 0.3, angle_text, transform=ax.transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

plt.tight_layout()
plt.show()



plt.savefig("Notes_Ekman_spiral/Figures/BVP_Prandt2.png",dpi=400)
plt.savefig("Notes_Ekman_spiral/Figures/nice figures/BVP_Prandtl_exp.png",dpi=400)