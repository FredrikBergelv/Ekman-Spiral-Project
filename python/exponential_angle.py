"""
Created on Fri May 22 20:10:17 2026

@author: fredrik
"""
import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt

f = 1e-4 
U0 = 1
k = 10 # do not need in reality 


def U(z_tilde, phi):
    "Nomrlaized z_tilde=zk"
    
    exp = np.exp(z_tilde/2)
    alpha = phi * (1+1j)
    num = exp*kv(1, alpha*exp)
    den = kv(1, alpha)
    ans = U0 * (1-num/den)
    return ans


def dUdz(z, phi):
    
    alpha = phi * (1+1j)
    num = kv(0, alpha * np.exp(k*z/2))
    den = 2*kv(1, alpha)
    ans = U0 * k * alpha * np.exp(k*z) * num /  den

    return ans

def ekman_transport(phi, zmax=10.0, Nz=50000):
    z = np.linspace(0, zmax, Nz)
    U_vals = U(z, phi)
    M = np.trapezoid(U_vals-U0, z)
    return M

def calculate_angle(dUdz):
    # Compute the angle of dU/dz
    theta = np.angle(dUdz, deg=True)
    return theta

extent = 4
phis = np.logspace(-50, np.log10(extent), 1000)

#%%
# ===============================
# Plotting
# ===============================
plt.figure(figsize=(8, 5))
plt.suptitle("Surface Angle for Exponetial Model", fontsize=14)
plt.title("Surface angle vs layer thickness",fontsize=13)

angles = np.angle(dUdz(0, phis), deg=True)

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


#%%


plt.figure(figsize=(8, 5))
plt.suptitle("Transport Angle for Exponetial Model", fontsize=14)
plt.title("Transport angle vs layer thickness",fontsize=13)

Ms = np.array([ekman_transport(phi) for phi in phis])
angles = np.angle(Ms, deg=True)

plt.hlines(135, min(phis), max(phis), color="black", linestyle='--', label="135° reference")
plt.plot(phis, angles)

plt.xlabel(r"Dimensionless layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Transport angle, $\theta_T$ [deg]",fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

plt.ylim(90,185)
plt.yticks([90, 105, 120, 135, 150, 165, 180])
plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
plt.legend(fontsize=11)

save_name="exponential_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()


