import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from kelvinfunctions_ker_kei import ker0, kei0, ker1, kei1

# Parameters
f = 1e-4  # Coriolis parameter
U0 = 1.0  # Reference velocity
k = 10 # do not need in reality 

def layer_dUdz0(phi, epsilon):
      R = (1 - epsilon) / (1 + epsilon)
      A1 = np.exp(phi)
      E1 = A1 * (np.cos(phi) + 1j * np.sin(phi))
      Z = (1 + 1j) * (E1**2 - R) / (E1**2 + R)
      return Z


def exp_dUdz(z, phi):
      arg1 = 2*phi*np.exp(k*z/2)
      arg2 = 2*phi
      num = ker0(arg1)+ 1j*kei0(arg1)
      denum = ker1(arg2) + 1j*kei1(arg2)
      ans = np.sqrt(1j)*U0*k*phi*np.exp(k*z)*(num /(denum) )
      return ans


# -------------------------
# Parameters
# -------------------------
epsilons = [0.0001, 0.01, 0.05, 0.1, 0.5]
phis = np.linspace(0,4,100000)




# Create figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(14,5), sharey=True, sharex=True)

# ===============================
# LEFT: 2-layer model
# ===============================
ax = axs[0]

for epsilon in epsilons:

    layer_angle = np.angle(layer_dUdz0(phis, epsilon), deg=True)

    ax.plot(phis, layer_angle, label=fr'$\epsilon={epsilon:.4f}$')

ax.hlines(45, min(phis), max(phis),
          linestyle='--', label="45° reference", color="black")
ax.vlines([np.pi/2, np.pi], -90, 90,
          linestyle=":", label=r"$\pi/2$, $\pi$", color="black")

ax.set_xlabel(r"Height ratio, $\varphi$ [-]")
ax.set_ylabel(r"Surface angle, $\theta$ [deg]")
ax.set_title("2-layer model")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=9)
ax.set_ylim(37, 93)

# ===============================
# RIGHT: Exponential model
# ===============================
ax = axs[1]

exp_angles = np.angle(exp_dUdz(0, phis), deg=True)
ax.plot(phis, exp_angles)

ax.hlines(45, min(phis), max(phis),
          linestyle='--', label="45° reference", color="black")

ax.set_xlabel(r"Height ratio, $\varphi$ [-]")
ax.set_title("Exponential viscosity model")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=9)
ax.set_ylim(37, 93)

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{x:.5f}".rstrip('0').rstrip('.'))
)

# ===============================
# Final layout
# ===============================
fig.suptitle("Surface angle vs layer thickness", fontsize=14)
plt.savefig("angle_comaprisson.png", dpi=400)
plt.tight_layout()
plt.show()
