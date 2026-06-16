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
  
def tau(z, phi, gamma, nu1=1):
    nu2 = gamma**2 * nu1
    hEK1 = np.sqrt(2*nu1/f)
    hEK2 = np.sqrt(2*nu2/f)
    H = phi*hEK1
    
    den = 1 - gamma + np.exp(2 * (1 + 1j) * phi) * (1 + gamma)
    A1 = -U0 * ((1 - gamma) / den)
    B1 =  U0 * ((1 + gamma) * np.exp(2 * (1 + 1j) * phi) / den)
    B2 =  U0 * (2 * np.exp((1 + 1j) * phi * (1 + 1/gamma)) / den)

    # tau = nu * dU/dz, with nu=1 for layer 1, nu=gamma^2 for layer 2
    tau1 =  nu1 * (1 + 1j)/hEK1 * A1 * np.exp( (1 + 1j) * z/hEK1) \
        + nu1 * (1 + 1j)/hEK1 * B1 * np.exp(-(1 + 1j) * z/hEK1)
    tau2 = nu2 * (1 + 1j)/hEK2 * B2 * np.exp(-(1 + 1j) * z/hEK2)

    return np.where(z < H, tau1, tau2)

def ekman_transport(phi, gamma):
    den = 1 - gamma + np.exp(2 * (1 + 1j) * phi) * (1 + gamma)
    # Only the complex structure matters for the angle
    tau0 = (1 + 1j) * (-(1 - gamma) + (1 + gamma) * np.exp(2 * (1 + 1j) * phi)) / den
    T = 1j * tau0  # i/f * tau0, f real so doesn't affect angle
    return np.angle(T, deg=True)



# Plottin gparameters
extent = 5

nu_ratios = np.array([0.001, 0.1, 0.5, 1/0.5, 1/0.1, 1/0.001])


phis =  np.logspace(-6, np.log10(extent), 1000)
phis =  np.linspace(0, extent, 1000)



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
ang_15_reversed = 2 * 45 - ang_15
plt.plot(phis, ang_15, label=r'1.5 model', color="black", linestyle='--')
plt.plot(phis, ang_15_reversed, color="black", linestyle='--')



plt.vlines([np.pi/2, np.pi], 5, 90, color="black", linestyle=":", label=r"$\pi/2$ and $\pi$")

plt.xlabel(r"Dimensionless lower layer thickness, $\varphi$ [-]",fontsize=11)
plt.ylabel(r"Surface angle, $\theta$ [deg]",fontsize=11)
plt.suptitle("Surface Angle for 2-layer Model", fontsize=14)
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

nu_ratios = [0.1, 10.0]
phis_to_plot = [0.2, 1.0, 4]


fig, axes = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex="row", sharey="row")

for i, nu_ratio in enumerate(nu_ratios):
    if nu_ratio == 0.1:
        nu1 = 1
    if nu_ratio == 10:
        nu1 = 0.1
    hEk1 = np.sqrt(2 * nu1 / f)

    zmax = 15 * hEk1
    Nz = 10000
    z = np.linspace(0, zmax, Nz)
    
    gamma = np.sqrt(nu_ratio)
    nu2 = gamma**2 * nu1
    hEk2 = np.sqrt(2 * nu2 / f)

    ax = axes[0, i]    

    for j, phi in enumerate(phis_to_plot):
        tau_vals = tau(z, phi, gamma, nu1=nu1)
        ax.plot(np.real(tau_vals), z, c=f"C{j}", label=fr'$\varphi={phi:.1f}$')
        ax.plot(np.imag(tau_vals), z, '--', c=f"C{j}")
                    
    xmin, xmax = ax.get_xlim()
    ax.fill_between([xmin, xmax], 0, hEk1, color='gray', alpha=0.2, label=r"$h_\text{Ek1}$", ec="black")
    ax.fill_between([xmin, xmax], hEk1, hEk1+hEk2, color='gray', alpha=0.1, label=r"$h_\text{Ek2}$", ec="black")
    
    ax.set_xlabel(r"Momentum flux, $\tau$ [m$^2$/s$^2$]", fontsize=11)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_title(fr"$\nu_2/\nu_1 = {nu_ratio}$ "
                 + (r"($\nu_2 < \nu_1$)" if gamma < 1 else r"($\nu_2 > \nu_1$)"),
                 fontsize=12)
    ax.set_ylim(0, zmax)
    ax.plot([], [], 'k-',  label=r'$\tau_x$')
    ax.plot([], [], 'k--', label=r'$\tau_y$')
    if i==1:
        ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    

    # --- Potential well below ---
    ax_pot = axes[1, i]
    for j, phi in enumerate(phis_to_plot):
        H = phi * hEk1  # interface height in metres
        nu_z = np.where(z < H, nu1, nu2)
        ax_pot.plot(f / nu_z, z, c=f"C{j}")
        ax_pot.axhline(H, color=f"C{j}", linestyle=':', linewidth=0.8)  # mark interface

    ax_pot.set_xlabel(r"Eigenvalue, $\lambda$ [m$^{-2}$]", fontsize=11)
    ax_pot.set_ylim(0, zmax)
    ax_pot.grid(True, linestyle='--', alpha=0.6)
    
axes[0, 0].set_ylabel(r"Height, $z$ [m]", fontsize=11)
axes[1, 0].set_ylabel(r"Height, $z$ [m]", fontsize=11)

plt.suptitle(r"Momentum Flux and Potential For 2-layer Model", fontsize=14)
plt.tight_layout()
save_name="2layer_structure"
plt.savefig(f"plots/{save_name}.png", dpi=400)
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400)
plt.show()