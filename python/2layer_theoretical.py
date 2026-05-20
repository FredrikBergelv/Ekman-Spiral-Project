"""
Created on Thu Feb 19 09:57:14 2026

@author: fredrik
"""
import numpy as np
import plot 

# -------------------------
# PARAMETERS
# -------------------------
u0      = 10.0        # Thermal-wind scale (m/s)
f       = 1e-4        # Coriolis parameter at 
z_max   = 400         # truncation<< depth (m)
z_surf  = 0.5         # At which height is surface? (for wind)
nz      = 5000        # grid points
z_heights = np.linspace(0.0, z_max, nz)

# Viscosity parameters
upper_viscosity = 0.1   # small (m2/s)
lower_viscosity = 1   # larger (m2/s)

h1 = np.sqrt(2 * lower_viscosity / f)
h2 = np.sqrt(2 * upper_viscosity / f)

H = 100
epsilon = h2 / h1


i = 1j   # imaginary unit

E1 = np.exp(H * (1 + i) / h1)
E2 = np.exp(H * (1 + i) / h2)

den = (1 - epsilon + E1**2 * (1 + epsilon))

A1 = -u0 * (1 - epsilon) / den
B1 = -u0 * E1**2 * (1 + epsilon) / den
B2 = -u0 * (2 * E1 * E2) / den

print("Parameters:")
epsilon = upper_viscosity / lower_viscosity
R = (1-epsilon) / (1+epsilon)
#print(f"   epsilon = {epsilon:.3f}")
print(f"   R = {R:.3f}")
print(f"   phi = {H/h1:.3f}")



# -------------------------
# ANALYTIC SOLUTION
# -------------------------
def analytic(z):
    z = np.asarray(z)

    U = np.zeros_like(z, dtype=complex)

    # Region 1 (z < H)
    mask1 = z < H
    U[mask1] = (
        A1 * np.exp(z[mask1] * (1 + i) / h1)
        + B1 * np.exp(-z[mask1] * (1 + i) / h1)
        + u0
    )

    # Region 2 (z >= H)
    mask2 = z >= H
    U[mask2] = (
        B2 * np.exp(-z[mask2] * (1 + i) / h2)
        + u0
    )

    return U.real, U.imag


# -------------------------
# PLOT
# -------------------------
u, v = analytic(z_heights)

theta = np.arctan2(v, u)        # in radians
theta_deg = np.degrees(theta)   # convert to degrees
print(f"   Surface angle: {theta_deg[1]:.2f} deg")
# Two-layer viscosity profile
nu_profile = np.where(z_heights < H, lower_viscosity, upper_viscosity)


# -------------------------
# Plotting
# -------------------------
import plot 
nu_profile = np.where(z_heights < H, lower_viscosity, upper_viscosity)
plot.subfig(z_heights, nu_profile, u, v, 
            savename="2layer_analytic_solution",
            title="2-layer eddy-viscosity: Theoretical solution") 


