import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv, kv

# ---------------------------------------------------------
# Parametrar
# ---------------------------------------------------------

epsilons = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15]

# Undvik exakt phi = 0 eftersom den analytiska lösningen
# innehåller division med phi.
phi1 = np.linspace(0, 4.0, 2000)
phi = phi1

# f*U_g påverkar inte argumentet om det är reellt och positivt
fUg = 1.0


# ---------------------------------------------------------
# tau(z) enligt randvillkoren
#
# tau'' = 2 i phi^2/(epsilon+z) * tau
#
# tau'(0) = -i f U_g
# tau(1)  = 0
# ---------------------------------------------------------

def tau(z, phi, epsilon, fUg=1.0):

    # Argumenten till Besselfunktionerna
    Xz = 2 * (1 + 1j) * phi * np.sqrt(epsilon + z)
    X0 = 2 * (1 + 1j) * phi * np.sqrt(epsilon)
    XH = 2 * (1 + 1j) * phi * np.sqrt(epsilon + 1.0)

    # Normaliseringsfaktor från tau'(0) = -i f U_g
    denominator = (
        iv(0, X0) * kv(1, XH)
        + kv(0, X0) * iv(1, XH)
    )

    C = (
        -(1 + 1j) * fUg
        / (2 * phi * denominator)
    )

    # Lösningen som automatiskt ger tau(1)=0
    result = (
        C
        * np.sqrt(epsilon + z)
        * (
            iv(1, Xz) * kv(1, XH)
            - kv(1, Xz) * iv(1, XH)
        )
    )

    return result


# ---------------------------------------------------------
# Plot arg(tau(z=0)) mot phi
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

for epsilon in epsilons:

    tau0 = tau(0.0, phi, epsilon, fUg)

    # Argument i radianer
    phase = np.angle(tau0, deg=True)

    plt.plot(
        phi,
        phase,
        label=fr"$\epsilon={epsilon:.0e}$"
    )

plt.xlabel(r"$\varphi$", fontsize=13)
plt.ylabel(r"$theta$, [deg]", fontsize=13)

plt.title(
    r"Analytic SUBC linear solution, $\tilde\nu=\epsilon+\tilde z$",
    fontsize=14
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()