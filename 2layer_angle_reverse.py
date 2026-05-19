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
epsilons = [1.1, 2, 5, 10, 100]
phis = np.linspace(0,4,100000)



plt.figure(figsize=(8,5))
for epsilon in epsilons:

    layer_angle = np.angle(layer_dUdz0(phis, epsilon), deg=True)

    plt.plot(phis, layer_angle, label=fr'$\epsilon={epsilon:.1f}$')

# -------------------------
# Reference lines
# -------------------------
plt.hlines(45, min(phis), max(phis), color="black", linestyle='--', label="45° reference")
plt.vlines([np.pi/2, np.pi], 30, 50, color="black", linestyle=":", label=r"$\pi/2$ and $\pi$")

plt.xlabel(r"Lower layer thickness, $\varphi$ [-]")
plt.ylabel(r"Surface angle, $\theta$ [deg]")
plt.suptitle("Surface Angle for 2-layer Model", fontsize=14)
plt.title("Surface angle vs layer thickness")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(0,55)
plt.savefig("2layer_angle_reverse.png", dpi=400)
plt.show()