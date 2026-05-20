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


# Plottin gparameters
epsilons = [0.0001, 0.01, 0.05, 0.1, 0.5]
phis = np.linspace(0,4,100000)



plt.figure(figsize=(8,5))
for epsilon in epsilons:

    layer_angle = np.angle(layer_dUdz0(phis, epsilon), deg=True)

    plt.plot(phis, layer_angle, label=fr'$\epsilon={epsilon:.4f}$')

# -------------------------
# Reference lines
# -------------------------
plt.hlines(45, min(phis), max(phis), color="black", linestyle='--', label="45° reference")
plt.vlines([np.pi/2, np.pi], 40, 90, color="black", linestyle=":", label=r"$\pi/2$ and $\pi$")

plt.xlabel(r"Lower layer thickness, $\varphi$ [-]")
plt.ylabel(r"Surface angle, $\theta$ [deg]")
plt.suptitle("Surface Angle for 2-layer Model", fontsize=14)
plt.title("Surface angle vs layer thickness")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(37,93)
plt.savefig("2layer_angle.png", dpi=400)
plt.show()

