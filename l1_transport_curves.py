import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Basic parameters
# ============================================================

kB = 8.617333262e-5      # Boltzmann constant, eV/K
T = 300                  # temperature, K

# 现在只是看曲线形状，所以 DOS prefactor 和 mobility 先设成 1
c = 1.0
mu = 1.0

# 因为最后会归一化，所以这里 e 也可以设为 1
e = 1.0


# Fermi-level range
EF_values = np.linspace(-0.2, 0.2, 401)

# Energy grid for numerical integration
Emax = 1.0
E = np.linspace(-Emax, Emax, 20001)


# ============================================================
# 2. Density of states: ideal graphene
#    D(E) ∝ |E|
# ============================================================

def DOS(E):
    return c * np.abs(E)


# ============================================================
# 3. Fermi-Dirac distribution
# ============================================================

def fermi(E, EF, T):
    return 1.0 / (1.0 + np.exp((E - EF) / (kB * T)))


# ============================================================
# 4. Band-edge / Dirac-point carrier counting
#
# electrons:
# n = integral from 0 to +infinity of D(E) f(E)
#
# holes:
# p = integral from -infinity to 0 of D(E) [1-f(E)]
# ============================================================

def carrier_band(EF):

    f = fermi(E, EF, T)

    positive = E >= 0
    negative = E <= 0

    n = np.trapezoid(
        DOS(E[positive]) * f[positive],
        E[positive]
    )

    p = np.trapezoid(
        DOS(E[negative]) * (1.0 - f[negative]),
        E[negative]
    )

    return n, p


# ============================================================
# 5. Fermi-referenced carrier counting
#
# electrons:
# n = integral from EF to +infinity
#
# holes:
# p = integral from -infinity to EF
# ============================================================

def carrier_fermi(EF):

    f = fermi(E, EF, T)

    above_EF = E >= EF
    below_EF = E <= EF

    n = np.trapezoid(
        DOS(E[above_EF]) * f[above_EF],
        E[above_EF]
    )

    p = np.trapezoid(
        DOS(E[below_EF]) * (1.0 - f[below_EF]),
        E[below_EF]
    )

    return n, p


# ============================================================
# 6. Transport model
# ============================================================

def transport(n, p, mu):

    RH = (p - n) / (e * (n + p)**2)

    Rxx = 1.0 / (e * mu * (n + p))

    return RH, Rxx


# ============================================================
# 7. Calculate all four curves
# ============================================================

RH_band = []
Rxx_band = []

RH_fermi = []
Rxx_fermi = []

for EF in EF_values:

    # Band-edge model
    n_band, p_band = carrier_band(EF)
    RH_b, Rxx_b = transport(n_band, p_band, mu)

    RH_band.append(RH_b)
    Rxx_band.append(Rxx_b)

    # Fermi-referenced model
    n_F, p_F = carrier_fermi(EF)
    RH_F, Rxx_F = transport(n_F, p_F, mu)

    RH_fermi.append(RH_F)
    Rxx_fermi.append(Rxx_F)


# Convert lists to arrays
RH_band = np.array(RH_band)
Rxx_band = np.array(Rxx_band)

RH_fermi = np.array(RH_fermi)
Rxx_fermi = np.array(Rxx_fermi)


# ============================================================
# 8. Normalise
#    现在主要比较 shape，而不是绝对数值
# ============================================================

RH_band_norm = RH_band / np.max(np.abs(RH_band))
RH_fermi_norm = RH_fermi / np.max(np.abs(RH_fermi))

Rxx_band_norm = Rxx_band / np.max(Rxx_band)
Rxx_fermi_norm = Rxx_fermi / np.max(Rxx_fermi)


# ============================================================
# 9. Plot 1: RH - EF, band-edge
# ============================================================

plt.figure()
plt.plot(EF_values, RH_band_norm)
plt.axhline(0)
plt.axvline(0, linestyle="--")
plt.xlabel("Fermi level $E_F$ (eV)")
plt.ylabel("Normalised $R_H$")
plt.title("$R_H$ vs $E_F$ - Band-edge counting")
plt.grid()
plt.show()


# ============================================================
# 10. Plot 2: Rxx - EF, band-edge
# ============================================================

plt.figure()
plt.plot(EF_values, Rxx_band_norm)
plt.axvline(0, linestyle="--")
plt.xlabel("Fermi level $E_F$ (eV)")
plt.ylabel("Normalised $R_{xx}$")
plt.title("$R_{xx}$ vs $E_F$ - Band-edge counting")
plt.grid()
plt.show()


# ============================================================
# 11. Plot 3: RH - EF, Fermi-referenced
# ============================================================

plt.figure()
plt.plot(EF_values, RH_fermi_norm)
plt.axhline(0)
plt.axvline(0, linestyle="--")
plt.xlabel("Fermi level $E_F$ (eV)")
plt.ylabel("Normalised $R_H$")
plt.title("$R_H$ vs $E_F$ - Fermi-referenced counting")
plt.grid()
plt.show()


# ============================================================
# 12. Plot 4: Rxx - EF, Fermi-referenced
# ============================================================

plt.figure()
plt.plot(EF_values, Rxx_fermi_norm)
plt.axvline(0, linestyle="--")
plt.xlabel("Fermi level $E_F$ (eV)")
plt.ylabel("Normalised $R_{xx}$")
plt.title("$R_{xx}$ vs $E_F$ - Fermi-referenced counting")
plt.grid()
plt.show()


# ============================================================
# 13. Physics checks
# ============================================================

print("\nPhysics checks:")

for test_EF in [0.0, 0.05, 0.073, 0.10, 0.20]:

    n_band, p_band = carrier_band(test_EF)
    n_F, p_F = carrier_fermi(test_EF)

    print(
        f"EF = {test_EF:.3f} eV : "
        f"n_band / n_Fermi = {n_band / n_F:.3f}"
    )