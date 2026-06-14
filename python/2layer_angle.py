"""
Created on Fri May 22 20:10:17 2026

@author: fredrik
"""
import numpy as np
import matplotlib.pyplot as plt


# Parameters
f = 1e-4  # Coriolis parameter
U0 = 1.0  # Reference velocity

  
def surface_angle(phi, gamma):        
      R = (1 - gamma) / (1 + gamma)
      num = np.exp((1+1j)*phi)-R*np.exp(-(1+1j)*phi)
      denum = np.exp((1+1j)*phi)+R*np.exp(-(1+1j)*phi)
      
      theta =  45 + np.angle(num/denum, deg=True)
      return theta
  
def F(z, phi, gamma, nu1=1):
    nu2 = gamma**2 * nu1
    hEK1 = np.sqrt(2*nu1/f)
    hEK2 = np.sqrt(2*nu2/f)
    H = phi*hEK1
    
    den = 1 - gamma + np.exp(2 * (1 + 1j) * phi) * (1 + gamma)
    A1 = -U0 * ((1 - gamma) / den)
    B1 =  U0 * ((1 + gamma) * np.exp(2 * (1 + 1j) * phi) / den)
    B2 =  U0 * (2 * np.exp((1 + 1j) * phi * (1 + 1/gamma)) / den)

    # F = nu * dU/dz, with nu=1 for layer 1, nu=gamma^2 for layer 2
    F1 =  nu1 * (1 + 1j)/hEK1 * A1 * np.exp( (1 + 1j) * z/hEK1) \
        + nu1 * (1 + 1j)/hEK1 * B1 * np.exp(-(1 + 1j) * z/hEK1)
    F2 = nu2 * (1 + 1j)/hEK2 * B2 * np.exp(-(1 + 1j) * z/hEK2)

    return np.where(z < H, F1, F2)

def ekman_transport(phi, gamma):
    den = 1 - gamma + np.exp(2 * (1 + 1j) * phi) * (1 + gamma)
    # Only the complex structure matters for the angle
    F0 = (1 + 1j) * (-(1 - gamma) + (1 + gamma) * np.exp(2 * (1 + 1j) * phi)) / den
    T = 1j * F0  # i/f * F0, f real so doesn't affect angle
    return np.angle(T, deg=True)



# Plottin gparameters
extent = 5

nu_ratios = np.array([0.001, 0.1, 0.5, 1/0.5, 1/0.1, 1/0.001])


phis =  np.logspace(-6, np.log10(extent), 1000)


#%%
plt.figure(figsize=(8*extent/4,5))
for ratio in nu_ratios:
    
    gamma = np.sqrt(ratio)

    layer_angle =surface_angle(phis, gamma)

    plt.plot(phis, layer_angle, label=fr'$\nu_2/\nu_1={gamma**2:.0e}$')


#plt.hlines(45, min(phis), max(phis), color="black", linestyle='--', label="45° reference")
ref = surface_angle(phis, 1)
plt.plot(phis, ref, label=r'$\nu_2/\nu_1=1$', color="black")

ang_15 = surface_angle(phis, 0)
ang_15_reversed = surface_angle(phis, 1e32)
plt.plot(phis, ang_15, label=r'1.5 model', color="black", linestyle='--')
plt.plot(phis, ang_15_reversed, color="black", linestyle='--')



plt.vlines([np.pi/2, np.pi], 5, 90, color="black", linestyle=":", label=r"$\pi/2$ and $\pi$")

plt.xlabel(r"Dimensionless lower layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
plt.suptitle("Surface Angle for 2-layer Model", fontsize=14)
plt.title("Surface angle vs lower layer thickness",fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="lower right", fontsize=11)
plt.ylim(0,95)
plt.yticks([0,15,30,45,60,75,90])
plt.xticks(np.arange(0, extent + 0.5, 0.5))
save_name="2layer_angle"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

#%%
phis =  np.logspace(-3, np.log10(extent), 1000)

plt.figure(figsize=(8*extent/4,5))
for ratio in nu_ratios:
    gamma = np.sqrt(ratio)
    
    trans_angle = np.array([ekman_transport(phi, gamma) for phi in phis])
    plt.plot(phis, trans_angle, label=fr'$\nu_2/\nu_1={gamma**2:.0e}$')

# -------------------------
# Reference lines
# -------------------------
plt.vlines([np.pi/2, np.pi], 95, 180, color="black", linestyle=":", label=r"$\pi/2$ and $\pi$")
plt.hlines(135, min(phis), max(phis), color="black", linestyle='--', label="135° reference")

ang_15 = ekman_transport(phis, 0)
ang_15_reversed = ekman_transport(phis, 1e32)
plt.plot(phis, ang_15, label=r'1.5 model', color="black", linestyle='--')
plt.plot(phis, ang_15_reversed, color="black", linestyle='--')

plt.xlabel(r"Dimensionless lower layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Transport angle, $\theta_T$ [deg]", fontsize=11)
plt.suptitle("Transport Angle for 2-layer Model", fontsize=14)
plt.title("Transport angle vs lower layer thickness",fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="lower right", fontsize=11)
plt.ylim(90,185)
plt.yticks([90, 105, 120, 135, 150, 165, 180])
plt.xticks(np.arange(0, extent + 0.5, 0.5))
save_name="2layer_transport"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)

plt.show()

#%%
nu1 = 0.10  # m^2/s, reference viscosity
hEk1 = np.sqrt(2 * nu1 / f)

nu_ratios = [0.1, 10.0]
phis_to_plot = [0.5, 1.0, 4]
zmax = 12 * hEk1
Nz = 10000
z = np.linspace(0, zmax, Nz)

fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex="row", sharey="row")

for i, nu_ratio in enumerate(nu_ratios):
    gamma = np.sqrt(nu_ratio)
    nu2 = gamma**2 * nu1
    ax = axes[0, i]

    for j, phi in enumerate(phis_to_plot):
        F_vals = F(z, phi, gamma, nu1=nu1)
        ax.plot(np.real(F_vals), z, c=f"C{j}", label=fr'$\varphi={phi:.1f}$')
        ax.plot(np.imag(F_vals), z, '--', c=f"C{j}")

    ax.set_xlabel(r"Vertical momentum flux, $F$ [m$^2$/s$^2$]", fontsize=11)
    ax.set_title(fr"$\nu_2/\nu_1 = {nu_ratio}$ "
                 + (r"($\nu_2 < \nu_1$)" if gamma < 1 else r"($\nu_2 > \nu_1$)"),
                 fontsize=12)
    ax.set_ylim(0, zmax)
    ax.plot([], [], 'k-',  label=r'$F_x$')
    ax.plot([], [], 'k--', label=r'$F_y$')
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Potential well below ---
    ax_pot = axes[1, i]
    for j, phi in enumerate(phis_to_plot):
        H = phi * hEk1  # interface height in metres
        nu_z = np.where(z < H, nu1, nu2)
        ax_pot.plot(1 / nu_z, z, c=f"C{j}", label=fr'$\varphi={phi:.1f}$')
        ax_pot.axhline(H, color=f"C{j}", linestyle=':', linewidth=0.8)  # mark interface

    ax_pot.set_xlabel(r"Potential well, $1/\nu$ [m$^{-2}$s]", fontsize=11)
    ax_pot.set_ylim(0, zmax)
    ax_pot.legend(loc="upper right", fontsize=9)
    ax_pot.grid(True, linestyle='--', alpha=0.6)
    
axes[0, 0].set_ylabel(r"Height, $z$ [m]", fontsize=11)
axes[1, 0].set_ylabel(r"Height, $z$ [m]", fontsize=11)

plt.suptitle(r"Vertical momentum flux and potential well for 2-layer viscosity", fontsize=14)
plt.tight_layout()
save_name="2layer_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()