#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:01:30 2025

@author: fredrik
"""
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

u0      = 1      # Thermal wind velocity
f       = 1      # The coriolis 
z_max   = 200    # Infinit height (choose several Ekman layers)
nz      = 10000  # Number of steps
z_surf  = 0.1        # At which height is surface? (for wind)


z_heights = np.linspace(0, z_max, nz)
z = sp.symbols('z')

#kin_visc = 1 
#kin_visc = 1 + 2/(1 + 3*z) + sp.sin(z/10)**3 * sp.exp(-0.03*z)  # Crazy
#kin_visc = 1 /(0.1+z)
#kin_visc = 1+z*sp.exp(-0.05*z) # Normal
#kin_visc = 1+10*sp.exp(-(z-100)**2/1000)  # Gaussian
kin_visc = 10+(1/0.1+z)*sp.sin(z)*sp.exp(-0.15*z)


### ================= THE FUNCTIONS ================= ####
def nu(z_val, z_sym=z, kin_visc=kin_visc):
    "This function describes the profile of the kinnematic viscosity"
    nu_func = sp.lambdify(z_sym, kin_visc, 'numpy')
    return nu_func(z_val) + 0*z_val # Ensures array like

def a_coupling(z, f=f):
    "This is the coupling coeficcent known as f/nu"
    nu_val = nu(z)
    coef = f/nu_val 
    return coef

def decay_rate(z, a=a_coupling):
    "This is the decay_rate 1/delta"
    coef = np.sqrt(a(z))/np.sqrt(2)
    return coef


# Convert to first-order system
# Y = [u, up, v, vp]
# Y' = [ up, -a(z)*v, vp, a(z)*u ]
def function(z, Y):
    """ This is the function we are trying to solve
        Convert to first-order system
        Y = [u, up, v, vp]
        Y' = [ up, -a(z)*v, vp, a(z)*u ] """
    u = Y[0]
    up = Y[1]
    v = Y[2]
    vp = Y[3]
    a = a_coupling(z)
    
    return np.vstack((up, -a * v, vp, a * u))


# Boundary conditions at z=0 and z=z_max
def bc(Y0, Yf):
    """ Boundary conditions at z=0 and z=z_max
        v(0)=0, u(0) = -u0
        at z_max: u' + k u = 0 and v' + k v = 0 """
        
    k_lim = decay_rate(z_max) # THis is the decay rate when z -> ininity
        
    u0_val, up0_val, v0_val, vp0_val = Y0
    uf_val, upf_val, vf_val, vpf_val = Yf
    
    return np.array([
        v0_val - 0,           # v(0) = 0
        u0_val + u0,            # u(0) = -u0  -> u(0)+u0 = 0    
        upf_val + k_lim * uf_val -k_lim * vf_val,   # u'(zmax) + k u(zmax) - k v(zmax)= 0
        vpf_val + k_lim * uf_val + k_lim * vf_val   # v'(zmax) + k u(zmax) + k v(zmax) = 0
        ])



### ================= NUMERIC CALCULATIONS ================= ####

# Check if converges
if not np.abs(nu(z_max*100)-nu(z_max))<0.01:
    raise ValueError(f'The kinematic viscoity was {nu(z_max):.1f} m²/s at {z_max} m, the value must converge!')

#Here we define the limit of the viscosity
nu_inft = nu(z_max)


# Make a guess of what decay rate should be if analytic
k_mean = 1/(np.sqrt(np.mean(a_coupling(z_heights)))/np.sqrt(2)) 


# Initial guess for solver (must be shape (4, z.size))
# Use analytic decaying guesses
u_guess = -u0 * np.exp(-k_mean*z_heights) * np.cos(k_mean*z_heights)  
v_guess =  u0 * np.exp(-k_mean*z_heights) * np.sin(k_mean*z_heights)
up_guess = np.gradient(u_guess, z_heights)
vp_guess = np.gradient(v_guess, z_heights)
Y_guess = np.vstack((u_guess, up_guess, v_guess, vp_guess))

# Solve BVP
sol = solve_bvp(function, bc, z_heights, Y_guess, max_nodes=20000)

if sol.status != 0:
    print("Warning: solve_bvp did not fully converge. status:", sol.status, "message:", sol.message)

#Extract the solution
u_sol = sol.y[0]
v_sol = sol.y[2]


# angle in radians
theta0 = np.arctan2(np.abs(v_sol[1]-v_sol[0]), np.abs(u_sol[1]-u_sol[0]))

# angle in degrees
theta0_deg = np.degrees(theta0)
print(f"Angle at the surface: {theta0_deg:.0f} degrees")



### ================= PLOTTING ================= ####

# Plots: depth profiles and top-down spiral
plt.close('all')
fig, axs = plt.subplots(1,3, figsize=(12,5))

#Here is the old analytic solution
u_old = -u0 * np.exp(-z_heights/np.sqrt(2 * f / nu_inft)) * np.cos(z_heights/np.sqrt(2 * f / nu_inft))
v_old = u0 * np.exp(-z_heights/np.sqrt(2 * f / nu_inft)) * np.sin(z_heights/np.sqrt(2 * f / nu_inft))

# Automatically generate LaTeX string
nu_latex = sp.latex(kin_visc)

plt.suptitle(rf"Kinematic viscosity: $\nu=$${nu_latex}$", fontsize=15)

ax = axs[0]
ax.plot(nu(z_heights), z_heights, color='green', label=r'$\nu=\nu(z)$')
ax.plot(nu_inft+0*z_heights, z_heights, '--', color='green', label=r'$\nu_\text{0}=\nu(z\longrightarrow \infty)$')
ax.set_xlabel(r"$\nu(z)$ [m²/s]", fontsize=12)
ax.set_ylabel("Height [m]", fontsize=12)
ax.set_title("Kinematic Viscosity Profile", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc="upper center", fontsize=12)
ax.text(0.97, 0.95, "(a)", transform=ax.transAxes, fontsize=12, fontname='DejaVu Sans', ha='right', va='top')


ax = axs[1]
ax.plot(u0+u_sol, z_heights, label=r"$u(z, \nu)$", c="C0")
ax.plot(u0+u_old, z_heights, label=r"$u(z, \nu_0)$", linestyle="--", c="C0")
ax.plot(v_sol, z_heights, label=r"$v(z, \nu)$", c="C1")
ax.plot(v_old, z_heights, label=r"$v(z, \nu_0)$",linestyle="--", c="C1")
ax.text(0.97, 0.95, "(b)", transform=ax.transAxes, fontsize=12, fontname='DejaVu Sans', ha='right', va='top')


ax.set_xlabel("velocity [m/s]", fontsize=12)
ax.set_ylabel("Height [m]", fontsize=12)
ax.set_ylim(0,20)
ax.set_title("Ekman Spiral vs Height", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)

ax = axs[2]
ax.plot(u0+u_sol, v_sol, '-k', label=r"$u(\nu)$ vs $v(\nu)$")
ax.plot(u0+u_old, v_old, '-k',linestyle="--", c="black", label=r"$u(\nu_0)$ vs $v(\nu_0)$")
ax.scatter(u0+u_sol[0], v_sol[0], color='red', label='surface')
ax.scatter(u0+u_sol[-1], v_sol[-1], color='green', label='top')
ax.set_xlabel("u [m/s]", fontsize=12)
ax.set_ylabel("v [m/s]", fontsize=12)
ax.set_title("Ekman Spiral Viewed from Above", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)
ax.axis('equal')
ax.text(0.97, 0.95, "(c)", transform=ax.transAxes, fontsize=12, fontname='DejaVu Sans', ha='right', va='top')


angle_text = f"Wind angle: {theta0_deg:.0f}°"
ax.text(0.05, 0.3, angle_text, transform=ax.transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

plt.tight_layout()
plt.show()

#plt.savefig("Notes_Ekman_spiral/Figures/BVP_fun.png",dpi=400)
#plt.savefig("Notes_Ekman_spiral/Figures/BVP_real.png",dpi=400)
#plt.savefig("Notes_Ekman_spiral/Figures/BVP_const.png",dpi=400)



