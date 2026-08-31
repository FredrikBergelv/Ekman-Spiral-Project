"""
Created on Fri May 22 20:10:17 2026

@author: fredrik
"""
import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt

f = 1e-4 
U0 = 10
k = 0.01 # do not need in reality 


def tau(z, phi, k=k):
    hek0 = (phi*k)**(-1)
    arg_z = 2 * (1 + 1j) * phi * np.exp(k * z / 2)
    arg_0 = 2 * (1 + 1j) * phi
    return ((1 + 1j) * f * U0 * hek0) / (2) * kv(0, arg_z) / kv(1, arg_0)


def surface_angle(phi):
    "Thus is the new Flux one"
    alpha = 4*phi * (1+1j)
    num = kv(0, alpha)
    den = kv(1, alpha)
    ans = (1+1j)*num /  den
    
    theta = np.angle(ans, deg=True)
    return theta

def ekman_transport(phi, k):
    T = (1j/f) * tau(0, phi, k=k)
    return T

extent = 4
phis = np.linspace(1e-50, extent, 1000)

#%%
# ============================================================
# Check the small-phi_exp limit
#
# For phi_exp << 1:
#
#   tau'' = 2 i k^2 phi_exp^2 exp(kz) tau
#
# Initially the curvature is small, so
#
#   tau'(z) ~ tau'(0) = -i U0
#
# and
#
#   tau(z) ~ tau(0) - i U0 z
#
# until approximately
#
#   z_star ~ (2/k) log(1/phi_exp)
#
# ============================================================

phi_exp = 0.1

h_ek0 = 100

# phi_exp = 1 / (k h_ek0)
k = 1 / (phi_exp * h_ek0)

z_1k = 1 / k

# Estimated end of approximately-linear region
z_star = (2 / k) * np.log(1 / phi_exp)

print("\n" + "=" * 70)
print("EXPONENTIAL MODEL: SMALL-PHI CHECK")
print("=" * 70)

print(f"phi_exp       = {phi_exp}")
print(f"h_Ek0         = {h_ek0:.6f}")
print(f"k             = {k:.6f}")
print(f"z_star        = {z_star:.6e}")



# ============================================================
# Function for numerical derivative
# ============================================================

def numerical_derivative(z, phi, dz):

    tau_plus = tau(z + dz, phi)
    tau_minus = tau(z - dz, phi)

    return (tau_plus - tau_minus) / (2 * dz)


# ============================================================
# Check at z = z_star
# ============================================================

z = z_star

tau_check = tau(z, phi_exp, k)

dz = z * 1e-4

tau_prime_check = numerical_derivative(
    z,
    phi_exp,
    dz
)


print("\n" + "-" * 70)
print("AT z = z_star")
print("-" * 70)

print(f"z             = {z:.6e}")

print(f"|tau(z_star)| = {abs(tau_check):.4e}")

print(f"|tau'(z_star)| = {abs(tau_prime_check):.4e}")

print(f"|tau(z_star)/tau'(z_star)| = {abs(tau_check/tau_prime_check):.4f}")


#%%
# ===============================
# Plotting surface angle
# ===============================

plt.figure(figsize=(8, 5))
plt.suptitle("Surface Angle for Exponetial Model", fontsize=14)

angles = surface_angle(phis)

plt.plot(phis, angles)

plt.xlabel(r"Dimensionless layer thickness, $\varphi_\text{exp}$ [-]",fontsize=11)
plt.ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

plt.hlines(45, min(phis), max(phis), color="black", linestyle='--', label="45° reference")
plt.legend(fontsize=11)

plt.ylim(0,95)
plt.yticks([0, 15, 30, 45, 60, 75, 90])
plt.xticks(np.arange(0, extent + 0.5, 0.5))

save_name="exponential_angle"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()


#%%%
# ===============================
# Plotting Transport 
# ===============================
k = 0.1 

plt.figure(figsize=(8, 5))
plt.suptitle("Transport for Exponetial Model", fontsize=14)

T = np.array([ekman_transport(phi, k) for phi in phis])
Tr, Ti = np.real(T), np.imag(T)
plt.plot(phis, Tr, color="C0", label=r"$T_x$")
plt.plot(phis, Ti, linestyle="--", color="C0", label=r"$T_y$")


plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Transport, $T$ [m$^2$/s]",fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

plt.yscale("symlog", linthresh=10)
plt.xticks(np.arange(0, extent + 0.5, 0.5))
plt.legend(fontsize=11)

save_name="exponential_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()

#%%
# ===============================
# Momentum flux
# ===============================

k_val = 0.01
phis_to_plot = [0.5, 1.0, 1.5]
Nz = 98
zmax = 5.2 * 1 / k_val
z = np.linspace(0, zmax, Nz)


fig, axes = plt.subplots(1, 1, figsize=(6, 5), sharex="row", sharey="row")

ax = axes
for j, phi in enumerate(phis_to_plot):
        
        nu0 = f / ( 2 * (phi*k_val)**2 )
        hEk = np.sqrt(2*nu0/f)
        tau_vals = tau(z, phi, k=k_val)
        ang = np.angle(tau_vals, deg=True)
        #ax.plot(ang, z, c=f"C{j}", label=fr'$\varphi={phi:.1f}$')
        ax.plot(np.real(tau_vals), z, c=f"C{j}", label=fr'$\varphi_\text{{exp}}={phi:.1f}$')
        ax.plot(np.imag(tau_vals), z, '--', c=f"C{j}")

xmin, xmax = ax.get_xlim()
for j, phi in enumerate(phis_to_plot):
        nu0 = f / ( 2 * (phi*k_val)**2 )
        print(nu0)
        hEk = np.sqrt(2*nu0/f)
        xaxis_old = 75  # 55 + 20
        new_xaxis = 0.12
        scaling_factor = new_xaxis / 70  # 0.02 / 70 ≈ 0.0002857

        position = [(j * 0.1 * xaxis_old + 55) * scaling_factor,
        (xaxis_old / 30 + j * 0.1 * xaxis_old + 55) * scaling_factor]
        
        ax.fill_between(position, 0, hEk, color=f"C{j}", alpha=0.5, ec="gray")   


ax.axhline(1/k_val, linestyle=":", c="black", label=r'$1/k$')
ax.fill_between([], [], alpha=0.5, color='gray', ec="gray", label=r"$h_\text{EK0}$")   

ax.set_xlabel(r"Momentum flux, $\tau$ [m$^2$/s$^2$]", fontsize=11)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylim(0, zmax)
ax.plot([], [], 'k-',  label=r'$\tau_x$')
ax.plot([], [], 'k--', label=r'$\tau_y$')
ax.legend(loc="upper center", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)


 
plt.suptitle(r"Momentum Flux for Exponential Model",
             fontsize=14)
plt.tight_layout()
save_name="exponential_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()

#%%

from scipy.integrate import cumulative_trapezoid
from matplotlib.colors import to_rgb


k_val = 0.01
phis_to_plot = [0.5, 1.0, 1.5]
Nz = 10000
zmax = 5.2 * 1 / k_val

z = np.linspace(0, zmax, Nz)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

for j, phi in enumerate(phis_to_plot):
    nu0 = f / ( 2 * (phi*k_val)**2 )
    print(nu0)
    tau_vals = tau(z, phi, k=k_val)
    nu_z = nu0 * np.exp(-k_val * z)
    U = cumulative_trapezoid(tau_vals/nu_z, z, initial=0)
    ang = np.angle(U, deg=True)   
    ax.plot(ang[1:], z[1:], c=f"C{j}", label=fr'$\varphi_\text{{exp}}={phi:.1f}$')
    
    integrand = tau_vals / nu_z
    

# --- Decorations: 1/k markers and hEk0 shading ---
for j, phi in enumerate(phis_to_plot):
    nu0 = f / ( 2 * (phi*k_val)**2 )
    hEk = np.sqrt(2*nu0/f)
    xaxis_old = 75  # 55 + 20
    new_xaxis = 75
    scaling_factor = new_xaxis / 70  
    position = [(j * 0.1 * xaxis_old + 55) * scaling_factor,
    (xaxis_old / 30 + j * 0.1 * xaxis_old + 55) * scaling_factor]
    ax.fill_between(position, 0, hEk, color=f"C{j}", alpha=0.5, ec="gray")  
    
    # Classical single-layer Ekman reference
    U_theory = U0 * (1 - np.exp(-(1 + 1j) * z / hEk))
    ang_theory = np.angle(U_theory, deg=True)
    color = np.array(to_rgb(f"C{j}")) * 0.5
    ax.plot(ang_theory[1:], z[1:], c=color, linestyle="--", lw=2)


# Legend proxies
ax.axhline(1/k_val, linestyle=":", c="black", label=r'$1/k$')
ax.fill_between([], [], alpha=0.5, color='gray', ec="gray", label=r"$h_\text{EK0}$")  
ax.plot([], [], 'k--', lw=2, label='classical solution')

ax.set_xlabel(r"Wind diection [°]", fontsize=11)
ax.set_ylabel(r"Height, $z$ [m]", fontsize=11)
ax.set_ylim(0, zmax)
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)

plt.suptitle(r"Spiral for Exponential Model", fontsize=14)

save_name = "exponential_angle_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()

