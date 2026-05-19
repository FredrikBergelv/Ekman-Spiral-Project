#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 22:20:55 2025

@author: fredrik
"""

import numpy as np
import matplotlib.pyplot as plt

u0     = 1      # thermal wind velocity
nu0    = 1      # viscosity
f      = 1      # Coriolis parameter
beta   = 0.15    # Kinematic viscosity chnage (per m) 
z_heights  = np.linspace(0, 20, 1000)


def kinematic_viscosity(z, nu0=nu0, beta=beta):
    nu = nu0 * np.exp(beta*z)
    return nu

nu = kinematic_viscosity(z_heights)
nu_mean = np.mean(nu)

def Ekman_layer_height(nu, f=f):
    h_Ek = np.sqrt(2 * nu / f)
    return h_Ek

h_Ek = Ekman_layer_height(nu0)

def Ekman_spiral(z, u0=u0, h_Ek=h_Ek):
    u_p  = -u0 * np.exp(-z / h_Ek) * np.cos( z / h_Ek)
    v    = u0 * np.exp(-z / h_Ek) * np.sin( z / h_Ek)
    return u_p, v

def prefactor(z, u0=u0, f=f, nu0=nu0, h_Ek=h_Ek, beta=beta):
    A0 = np.exp(-f*h_Ek / (2*nu0 * beta))
    A = A0 * np.exp( z/h_Ek + (f*h_Ek/(2*nu0*beta)) * np.exp(-beta*z))
    return A

def new_Ekman_spiral(z):
    NEkS = prefactor(z)*Ekman_spiral(z)
    return NEkS


u_p_new, v_new = new_Ekman_spiral(z_heights)
u_p, v = Ekman_spiral(z_heights)



#%%%

fig, axes = plt.subplots(1, 3, figsize=(18,6))
ax = axes[0]
ax.plot(nu, z_heights, color='green', label=r'$\nu=\nu_0e^{\beta z}$')
ax.plot(nu0+0*z_heights, z_heights, '--', color='green', label=r'$\nu_\text{0}$')
ax.set_xlabel(r"$\nu(z)$ [m²/s]", fontsize=12)
ax.set_ylabel("Height [m]", fontsize=12)
ax.set_title("Kinematic Viscosity Profile", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)

ax = axes[1]
ax.plot(u0 + u_p_new, z_heights, label=r"$\tilde u'(z)$", color='C0')
ax.plot(u0 + u_p, z_heights, label=r"$u'(z)$", linestyle="--", color='C0')
ax.plot(v_new, z_heights, label=r"$\tilde v(z)$", color='C1')
ax.plot(v, z_heights, label=r"$v(z)$", linestyle="--", color='C1')
ax.set_xlabel("Velocity [m/s]", fontsize=12)
ax.set_ylabel("Height [m]", fontsize=12)
ax.set_title("Ekman Spiral vs Height", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)

ax = axes[2]
ax.plot(u0 + u_p, v, '--', color='black', label='u vs v')
ax.plot(u0 + u_p_new, v_new, '-', color='black', label=r'$\tilde u$ vs $\tilde v$')
ax.scatter(u0+u_p_new[0], v_new[0], color='red', label='surface')
ax.scatter(u0+u_p_new[-1], v_new[-1], color='green', label='top')
ax.set_xlabel("u [m/s]", fontsize=12)
ax.set_ylabel("v [m/s]", fontsize=12)
ax.set_title("Ekman Spiral Viewed from Above", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)
ax.axis('equal')



plt.tight_layout()
plt.show()

#%%

fig, axes = plt.subplots(1,2, figsize=(12,5))

ax = axes[0]
ax.plot(u0 + u_p_new, z_heights, label=r"$\tilde u'(z)$", color='C0')
ax.plot(u0 + u_p, z_heights, label=r"$u'(z)$", linestyle="--", color='C0')
ax.plot(v_new, z_heights, label=r"$\tilde v(z)$", color='C1')
ax.plot(v, z_heights, label=r"$v(z)$", linestyle="--", color='C1')
ax.set_xlabel("Velocity [m/s]", fontsize=12)
ax.set_ylabel("Height [m]", fontsize=12)
ax.set_title("Ekman Spiral vs Height", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)

ax = axes[1]
ax.plot(u0 + u_p, v, '--', color='black', label='u vs v')
ax.plot(u0 + u_p_new, v_new, '-', color='black', label=r'$\tilde u$ vs $\tilde v$')
ax.scatter(u0+u_p_new[0], v_new[0], color='red', label='surface')
ax.scatter(u0+u_p_new[-1], v_new[-1], color='green', label='top')
ax.set_xlabel("u [m/s]", fontsize=12)
ax.set_ylabel("v [m/s]", fontsize=12)
ax.set_title("Ekman Spiral Viewed from Above", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)
ax.axis('equal')

plt.tight_layout()
plt.show()

plt.savefig(f"Notes_Ekman_spiral/Figures/analytic_b={beta}.png",dpi=400)



#%%

def dx_C(z):
    return f * beta * np.exp(-beta*z) / nu0

def C(z):
    return (f * np.exp(-beta*z) / nu0) ** (3/2)

plt.title(r"Check that $\frac{\partial C(z)}{\partial z} \ll  C^{3/2} $", fontsize=14)
plt.plot(z_heights, dx_C(z_heights), label=r"$\frac{\partial C(z)}{\partial z}$")
plt.plot(z_heights, C(z_heights), label=r"$C^{3/2}$")
plt.grid()
plt.legend(fontsize=12)
plt.show()
