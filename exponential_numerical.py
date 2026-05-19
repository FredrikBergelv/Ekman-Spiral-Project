"""
Created on Mon Mar  2 22:00:51 2026

@author: fredrik
"""

import numpy as np
from scipy.integrate import solve_bvp
from plot import subfig 

# -------------------------
# Parameters / grid
# -------------------------
u0      = 10.0            # Thermal-wind scale (m/s)
f       = 1e-4            # Coriolis parameter at 
z_max   = 400             # truncation depth (m)
z_surf  = 0.5             # At which height is surface? (for wind)
nz      = 5000            # grid points
z_heights = np.linspace(0.0, z_max, nz)

# Viscosity parameters


#-------------
# nu(z): Prandtl eddy viscosity (vectorized)
# -------------------------
def nu(z, k=0.03, A=1, B=0):
    """
    exponenial function
    """
    z = np.asarray(z)

    nu_array = A * np.exp(-k*z) + B

    return nu_array




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
    u     = Y[0]
    tau_u = Y[1]
    v     = Y[2]
    tau_v = Y[3]

    nu_val = nu(z)

    return np.vstack((
        tau_u / nu_val,     # u'
        -f * v,             # tau_u'
        tau_v / nu_val,     # v'
        f * u               # tau_v'
    ))

# -------------------------
# Boundary conditions
# -------------------------
def bc(Y0, Yf):
    u0_val, tau_u0, v0_val, tau_v0 = Y0
    uf_val, tau_uf, vf_val, tau_vf = Yf

    return np.array([
        u0_val + u0,   # u(0) = -u0
        v0_val,        # v(0) = 0
        tau_uf,        # stress → 0 at top
        tau_vf
        ])


# -------------------------
# initial guess (physically motivated)
# -------------------------
# estimate mean k from mean a
k_mean = np.sqrt(np.mean(a_coupling(z_heights))) / np.sqrt(2.0)

u_guess  = -u0*np.exp(-k_mean*z_heights)*np.cos(k_mean*z_heights)
up_guess =  u0*np.exp(-k_mean*z_heights)*(k_mean*np.cos(k_mean*z_heights) + k_mean*np.sin(k_mean*z_heights))
tau_u_guess = nu(z_heights)  
tau_v_guess = nu(z_heights) 

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

u_sol = sol.y[0] + u0
v_sol = sol.y[2] 
z_sol = sol.x


# -------------------------
# Angle at surface
# -------------------------
idx = np.argmin(np.abs(z_sol - z_surf))
theta0 = np.arctan2(np.abs(v_sol[idx]-v_sol[0]), np.abs(u_sol[idx]-u_sol[0]))
theta_deg = np.degrees(theta0)
print(f"Angle at the surface: {theta_deg:.0f} degrees")
print("\n")


# -------------------------
# Plotting
# -------------------------
nu_profile = nu(z_sol)
subfig(z_sol, nu_profile, u_sol, v_sol, 
       savename="exponential_numerical_solution",
       title="exponetial eddy-viscosity: Numerical solution",
       nu_version = r"$\nu=A e^{-kz}$")




