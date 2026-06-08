"""
Created on Fri May 22 20:10:17 2026

@author: fredrik
"""
import numpy as np
import matplotlib.pyplot as plt


# Parameters
f = 1e-4  # Coriolis parameter
U0 = 1.0  # Reference velocity

def layer_dUdz0(phi, epsilon):
      R = (1 - epsilon) / (1 + epsilon)
      A1 = np.exp(phi)
      E1 = A1 * (np.cos(phi) + 1j * np.sin(phi))
      Z = (1 + 1j) * (E1**2 - R) / (E1**2 + R)
      return Z
  
def U(z_tilde, phi, epsilon):
    "This solution is normalizd for h_1"

    gamma = epsilon
    den = 1 - gamma + np.exp((1 + 1j)*phi) * (1 + gamma)
    A1 = -U0 * ((1 - gamma) / den)
    B1 = -U0 * ((1 + gamma) * np.exp((1 + 1j)*phi) / den)
    B2 = -U0 * (2* np.exp(((1 + 1j)) * phi * (1 + 1/gamma))/ den)

    U1 =  U0 + A1 * np.exp((1 + 1j)*z_tilde) + B1 * np.exp(-(1 + 1j)*z_tilde)
    U2 =  U0 + B2 * np.exp(-(1 + 1j)*z_tilde/gamma)

    return np.where(z_tilde < phi, U1, U2)


def ekman_transport(phi, epsilon, zmax=1500.0, Nz=500000):
    z = np.linspace(0, zmax, Nz)
    U_vals = U(z, phi, epsilon)
    M = np.trapezoid(U_vals-U0, z)
    print(np.angle(M, deg=True), np.angle(M, deg=True)-np.angle(layer_dUdz0(phi, epsilon), deg=True))
    return M

def one_and_half_model(phis):
    from scipy.integrate import solve_bvp
    u0=10
    
    def solve_profile(phi):

        z = np.linspace(0, 1, 300)
        
        def fun(z, Y):
            u, up, v, vp = Y
            
            nu_z = 1
            nu_zp = 0

            return np.vstack([
                up,
                (-2*phi**2*v - nu_zp*up) / nu_z,
                vp,
                ( 2*phi**2*(u - u0) - nu_zp*vp) / nu_z
            ])

        def bc(Y0, YH):
            return np.array([
                Y0[0],   # u_r(0)=0
                Y0[2],   # u_i(0)=0
                YH[1],   # u_r'(H)=0
                YH[3]    # u_i'(H)=0
            ])

        # initial guess (smooth decay)
        k = np.sqrt(phi**2 + 1e-6)
        u_hat = u0 * (1 - np.exp(-k*z))
        v_hat = np.zeros_like(z)

        Y0 = np.vstack([
            u_hat,
            np.gradient(u_hat, z),
            v_hat,
            np.gradient(v_hat, z)])

        sol = solve_bvp(fun, bc, z, Y0)

        return sol.x, sol.y

    def surface_angle(z, Y):
        u, up, v, vp = Y

        idx = 1 # surface 

        angle = np.arctan2(vp[idx], up[idx])
        return np.degrees(angle)
    
    angs = []
    for phi in phis:
        solh1, solh2 = solve_profile(phi)
        surf = surface_angle(solh1, solh2)
        angs.append(surf)

    return angs


# Plottin gparameters
extent = 4

nu_ratios = np.array([0.001, 0.1, 0.5, 1/0.5, 1/0.1, 1/0.001])
epsilons = np.sqrt(nu_ratios)

phis = np.linspace(0,extent,10000)

#%%
plt.figure(figsize=(8*extent/3,5))
for epsilon in epsilons:

    layer_angle = np.angle(layer_dUdz0(phis, epsilon), deg=True)

    plt.plot(phis, layer_angle, label=fr'$\nu_2/\nu_1={epsilon**2:.0e}$')


#plt.hlines(45, min(phis), max(phis), color="black", linestyle='--', label="45° reference")
ref = np.angle(layer_dUdz0(phis, 1), deg=True)
plt.plot(phis, ref, label=r'$\nu_2/\nu_1=1$', color="black", linestyle='--')

phis_15 = np.logspace(-6, np.log10(extent), 10000)
plt.plot(phis_15, one_and_half_model(phis_15), label=r'1.5 model', color="black")


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
phis = np.linspace(0,extent,300)
nu_ratios = np.array([0.00001, 0.1, 0.5, 1/0.5, 1/0.1, 1/0.00001])
epsilons = np.sqrt(nu_ratios)

plt.figure(figsize=(8*extent/3,5))
for epsilon in epsilons:

    M_vals = np.array([ekman_transport(phi, epsilon) for phi in phis])
    trans_angle = np.angle(M_vals, deg=True)
    plt.plot(phis, trans_angle, label=fr'$\nu_2/\nu_1={epsilon**2:.0e}$')

# -------------------------
# Reference lines
# -------------------------
plt.vlines([np.pi/np.sqrt(2), np.sqrt(2)*np.pi], 90, 185, color="black", linestyle=":", label=r"$\pi/\sqrt{2}$ and $\sqrt{2}\pi$")
plt.hlines(135, min(phis), max(phis), color="black", linestyle='--', label="135° reference")

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

