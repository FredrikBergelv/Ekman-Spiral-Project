import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------
u0 = 10.0
f = 1e-4
min_viscosities = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15]

# -------------------------
# VISCOSITY SCHEME
# -------------------------
def nu_parabolic(z, eps):
    return 4*z*(1 - z) + eps

# -------------------------
# SOLVER
# -------------------------
def solve_profile(phi, eps, prev_sol):
    def fun(z, Y):
        taux, taupx, tauy, taupy = Y
        nu = nu_parabolic(z, eps)
        return np.vstack([
            taupx,
            -2*phi**2*tauy / nu,
            taupy,
             2*phi**2*taux / nu
        ])

    def bc(Y0, Y1):
        return np.array([
            Y1[0],
            Y1[2],
            Y0[1],
            Y0[3] + f*u0,
        ])

    if prev_sol is not None:
        z0 = prev_sol.x
        Y0 = prev_sol.y
    else:
        z0 = np.linspace(0, 1, 300)
        nu_mid = nu_parabolic(0.5, eps)
        h_Ek = np.sqrt(2 * nu_mid / f)
        exp_decay = np.exp(-z0 / h_Ek)
        Fx_g  =  (nu_mid * u0 / h_Ek) * exp_decay * np.cos(z0 / h_Ek)
        Fy_g  = -(nu_mid * u0 / h_Ek) * exp_decay * np.sin(z0 / h_Ek)
        Fpx_g =  (nu_mid * u0 / h_Ek**2) * exp_decay * (-np.cos(z0/h_Ek) - np.sin(z0/h_Ek))
        Fpy_g =  (nu_mid * u0 / h_Ek**2) * exp_decay * (-np.sin(z0/h_Ek) + np.cos(z0/h_Ek))
        Y0 = np.vstack([Fx_g, Fpx_g, Fy_g, Fpy_g])

    sol = solve_bvp(fun, bc, z0, Y0, tol=1e-8, max_nodes=13000)
    return sol

def surface_angle(sol):
    Y0 = sol.sol(0.0)
    taux0, _, tauy0, _ = Y0
    return np.degrees(np.arctan2(tauy0, taux0))

def transport(sol):
    Y0 = sol.sol(0.0)
    taux0, _, tauy0, _ = Y0
    Tx = -(1/f )* tauy0
    Ty = (1/f )* taux0

    return [Tx, Ty]


# -------------------------
# COMPUTE
# -------------------------
points = 100

phi_zoom   = np.linspace(0.01, 4,   points)
phi_wide   = np.linspace(4,    250, points)
phi_wide   = np.logspace(np.log(4), 5, 40)

phi_values = np.concatenate([phi_zoom, phi_wide])

surf_results  = {eps: [] for eps in min_viscosities}
T_results  = {eps: [] for eps in min_viscosities}


for j, epsilon in enumerate(min_viscosities):
    prev_sol = None
    for i, phi in enumerate(phi_values):
        sol = solve_profile(phi, epsilon, prev_sol)
        
        surf_results[epsilon].append(surface_angle(sol))
        T_results[epsilon].append(transport(sol))

        if sol.success:
            prev_sol = sol
        else:
            prev_sol = None

        percent = 100 * (j * len(phi_values) + i + 1) / (len(phi_values) * len(min_viscosities))
        print(f"{percent:.1f}%  phi={phi:.2f}  eps={epsilon:.0e}"
              f"  surf={surf_results[epsilon][-1]:.2f}")

mask_zoom = phi_values <= 4
mask_wide = phi_values >= 4

#%%
# ===============================
# Plotting surface angle
# ===============================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True,
                                gridspec_kw={'width_ratios': [8, 4], 'wspace': 0.05})

for j, epsilon in enumerate(min_viscosities):
    angles = np.array(surf_results[epsilon])
    label  = fr"$\epsilon={epsilon:.0e}$"
    ax1.plot(phi_values[mask_zoom], angles[mask_zoom], label=label, c=f"C{j}")
    ax2.plot(phi_values[mask_wide], angles[mask_wide],  label=label, c=f"C{j}", alpha=0.6)

for ax in (ax1, ax2):
    ax.axhline(45, color="black", linestyle='--', linewidth=1, label="45° reference")
    ax.set_ylim(0, 95)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])

ax1.grid(True, linestyle='--', alpha=0.6)
ax2.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(0, 4)
#ax2.set_xlim(4, 250)
ax1.set_xticks(np.arange(0, 4.5, 0.5))
#ax2.set_xticks([50, 100, 150, 200, 250])
ax2.set_xscale("log")
plt.setp(ax2.get_xticklabels(), style='italic')
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(left=False)   

fig.text(0.5, 0.03, r"Dimensionless layer thickness, $\varphi$ [-]",
         ha='center', fontsize=11)
ax1.set_xlabel("")
ax2.set_xlabel("")
plt.subplots_adjust(bottom=0.13)

ax1.set_ylabel(r"Surface angle, $\theta$ [deg]", fontsize=11)
ax2.legend(fontsize=11, loc="upper right")
fig.suptitle("Surface Angle for SUBC Parabolic Model", fontsize=14)

save_name = "numerical_SUBC_parabolic_angle_broken"
plt.savefig(f"plots/{save_name}.png", dpi=400, bbox_inches='tight')
plt.savefig(f"../Ekman-Spirals-with-Variable-Eddy-Viscosity-Article/Figures/{save_name}.png", dpi=400, bbox_inches='tight')
plt.show()

#%%

# ===============================
# Plotting Transport 
# ===============================
