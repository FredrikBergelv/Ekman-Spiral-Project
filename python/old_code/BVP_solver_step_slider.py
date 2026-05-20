#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 09:07:10 2025

@author: fredrik
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_bvp

# -------------------------
# Parameters / grid
# -------------------------
u0      = 5.0
Omega   = 2*np.pi / (24*3600)
lat_rad = np.deg2rad(45.0)
f       = 2 * Omega * np.sin(lat_rad)

z_max   = 400
nz      = 5000
z_heights = np.linspace(0.0, z_max, nz)
z_surf = 0.1

background_viscosity = 0.4


# -----------------------------------------
# nu(z) with midpoint, thickness, increase
# -----------------------------------------
def nu(z, mid, d, inc):
    L1 = mid - d
    L2 = mid + d
    z = np.asarray(z)

    mask = (z >= L1) & (z <= L2)
    nu_array = np.full_like(z, background_viscosity, dtype=float)
    nu_array[mask] += inc

    if nu_array.size == 1:
        return float(nu_array)
    return nu_array


# -------------------------
# coupling a(z)
# -------------------------
def a_coupling(z, mid, d, inc):
    return f / nu(z, mid, d, inc)


def decay_rate(z, mid, d, inc):
    a = a_coupling(z, mid, d, inc)
    return np.sqrt(a) / np.sqrt(2)


# -------------------------
def fun(z, Y, mid, d, inc):
    u, up, v, vp = Y
    a = a_coupling(z, mid, d, inc)
    return np.vstack((up,
                      -a * v,
                      vp,
                      a * u))


def bc(Y0, Yf, mid, d, inc):
    k_lim = decay_rate(z_max, mid, d, inc)
    u0v, up0, v0v, vp0 = Y0
    uf, upf, vf, vpf = Yf
    return np.array([
        v0v,
        u0v + u0,
        upf + k_lim * uf,
        vpf + k_lim * vf
    ])


# ------------- solve ekman -------------
def solve_ekman(mid, d, inc):
    k_mean = np.sqrt(np.mean(a_coupling(z_heights, mid, d, inc))) / np.sqrt(2)
    u_guess = -u0 * np.exp(-k_mean * z_heights) * np.cos(k_mean * z_heights)
    v_guess =  u0 * np.exp(-k_mean * z_heights) * np.sin(k_mean * z_heights)
    up_guess = np.gradient(u_guess, z_heights)
    vp_guess = np.gradient(v_guess, z_heights)
    Y_guess = np.vstack((u_guess, up_guess, v_guess, vp_guess))

    def fun_local(z, Y):
        return fun(z, Y, mid, d, inc)

    def bc_local(Y0, Yf):
        return bc(Y0, Yf, mid, d, inc)

    sol = solve_bvp(fun_local, bc_local, z_heights, Y_guess, max_nodes=20000)
    return sol


# -------------------------
# Create figure + sliders
# -------------------------
fig, axs = plt.subplots(1,3, figsize=(14,6))
fig.suptitle(r'Step eddy-viscosity: $\nu(z)=$ ${\chi}_\text{BL}(z)$ + $\nu_\text{karman}$', fontsize=15)
plt.subplots_adjust(bottom=0.30)

# Initial slider values
mid_0 = 100
d_0   = 20
inc_0 = 1.0

sol = solve_ekman(mid_0, d_0, inc_0)
u_sol = sol.y[0]
v_sol = sol.y[2]
z_sol = sol.x

# Compute initial angle
idx = np.argmin(np.abs(z_sol - z_surf))
theta0 = np.arctan2(np.abs(v_sol[idx]-v_sol[0]), np.abs(u_sol[idx]-u_sol[0]))
theta_deg = np.degrees(theta0)

# -------------------------
# Plot initial fields
# -------------------------

a_const = f / background_viscosity
k_const = np.sqrt(a_const) / np.sqrt(2.0)

u_old = -u0 * np.exp(-k_const * z_sol) * np.cos(k_const * z_sol)
v_old =  u0 * np.exp(-k_const * z_sol) * np.sin(k_const * z_sol)

ax1 = axs[0]
line_nu, = ax1.plot(nu(z_sol, mid_0, d_0, inc_0), z_sol, label=r'$\nu_\text{step}(z)$', color='green')
ax1.plot(background_viscosity+0*z_sol, z_sol, '--', color='green', label=r'$\nu_\text{karman}$')
ax1.set_title("Kinematic Viscosity Profile", fontsize=13)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.text(0.97, 0.95, "(a)", transform=ax1.transAxes, fontsize=12, fontname='DejaVu Sans', ha='right', va='top')
ax1.set_xlabel(r"$\nu(z)$ [m²/s]", fontsize=12)
ax1.set_ylabel("Height [m]", fontsize=12)
ax1.legend(loc="upper center", fontsize=12)

ax2 = axs[1]
line_u, = ax2.plot(u0+u_sol, z_sol, label=r"$u(z, \nu_\text{step})$")
line_v, = ax2.plot(v_sol, z_sol, label=r"$v(z, \nu_\text{step})$")
ax2.set_title("Ekman Spiral vs Height", fontsize=13)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.text(0.97, 0.95, "(b)", transform=ax2.transAxes, fontsize=12, fontname='DejaVu Sans', ha='right', va='top')
ax2.set_xlabel("velocity [m/s]", fontsize=12)
ax2.set_ylabel("Height [m]", fontsize=12)
ax2.plot(v_old, z_sol, label=r"$v(z, \nu_\text{kar})$",linestyle="--", c="C1")
ax2.plot(u0+u_old, z_sol, label=r"$u(z, \nu_\text{kar})$", linestyle="--", c="C0")
ax2.legend(loc="upper center", fontsize=12)

ax3 = axs[2]
line_spiral, = ax3.plot(u0+u_sol, v_sol, '-k', label=r"$u(\nu_\text{step})$ vs $v(\nu_\text{step})$")
ax3.set_title("Ekman Spiral Viewed from Above", fontsize=13)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.set_xlabel("u [m/s]", fontsize=12)
ax3.set_ylabel("v [m/s]", fontsize=12)
ax3.axis('equal')
ax3.plot(u0+u_old, v_old, '-k',linestyle="--", c="black", label=r"$u(\nu_\text{kar})$ vs $v(\nu_\text{kar})$")
ax3.scatter(u0+u_sol[0], v_sol[0], color='red', label='surface')
ax3.scatter(u0+u_sol[-1], v_sol[-1], color='green', label='top')
ax3.text(0.97, 0.95, "(c)", transform=ax3.transAxes, fontsize=12, fontname='DejaVu Sans', ha='right', va='top')
ax3.legend(loc="upper left", fontsize=12)

# -------------------------
# PERSISTENT TEXTBOX (important part)
# -------------------------
angle_text = ax3.text(
    0.05, 0.30,
    f"Wind angle: {theta_deg:.0f}°",
    transform=ax3.transAxes,
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
)


# -------------------------
# Sliders
# -------------------------
ax_mid = plt.axes([0.3, 0.15, 0.4, 0.04])
ax_d   = plt.axes([0.3, 0.10, 0.4, 0.04])
ax_inc = plt.axes([0.3, 0.05, 0.4, 0.04])

slider_mid = Slider(ax_mid, 'Midpoint [m]', 0, 400, valinit=mid_0)
slider_d   = Slider(ax_d,   'Thickness [m]', 0, 400, valinit=d_0)
slider_inc = Slider(ax_inc, 'Step size [m²/s]', 0, 10, valinit=inc_0)


# -------------------------
# Update function
# -------------------------
def update(val):
    mid = slider_mid.val
    d   = slider_d.val
    inc = slider_inc.val

    sol = solve_ekman(mid, d, inc)
    u = sol.y[0]
    v = sol.y[2]

    # Update curves
    line_nu.set_xdata(nu(z_sol, mid, d, inc))
    line_u.set_xdata(u0 + u)
    line_v.set_xdata(v)
    line_spiral.set_xdata(u0 + u)
    line_spiral.set_ydata(v)

    # Update angle textbox (persistent)
    idx = np.argmin(np.abs(z_sol - z_surf))
    theta0 = np.arctan2(np.abs(v[idx]-v[0]), np.abs(u[idx]-u[0]))
    theta_deg = np.degrees(theta0)
    angle_text.set_text(f"Wind angle: {theta_deg:.0f}°")

    fig.canvas.draw_idle()


slider_mid.on_changed(update)
slider_d.on_changed(update)
slider_inc.on_changed(update)

plt.show()
