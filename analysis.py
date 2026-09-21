from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

base = Path(__file__).resolve().parent
l1 = pd.read_csv(base / "L1_data.csv")

with open(base / "L1_params.json", "r", encoding="utf-8") as f:
    params = json.load(f)

with open(base / "results_template.json", "r", encoding="utf-8") as f:
    results = json.load(f)

kB = 8.617333e-5
NA = 3.816e15
e = 1.602176634e-19
Emax = 2.5

c = params["c"]
T = params["T"]
EF_op = params["EF_op_eV"]
rho = params["rho_ohm_sq"]

def fermi(E, EF, T):
    x = np.clip((E - EF) / (kB * T), -60, 60)
    return 1.0 / (np.exp(x) + 1.0)

def n_band(EF, T, c):
    E = np.linspace(0.0, Emax, 20001)
    return NA * np.trapezoid(c * E * fermi(E, EF, T), E)

def n_fermi(EF, T, c):
    E = np.linspace(EF, Emax, 20001)
    return NA * np.trapezoid(c * E * fermi(E, EF, T), E)

def n_fermi_closed(EF, T, c):
    kT = kB * T
    return NA * c * kT * (EF * np.log(2) + (np.pi**2 / 12) * kT)

# Gate checks
print("=== L1 physics gate ===")
for EF_test, expected in [(0.0,1.00),(0.073,2.00),(0.100,2.60),(0.200,5.10)]:
    ratio = n_band(EF_test, T, c) / n_fermi(EF_test, T, c)
    print(f"EF={EF_test:.3f} eV: C={ratio:.3f} (expected ~{expected:.2f})")

# Sweep
l1["n_fermi_cm2"] = [n_fermi(EF, T, c) for EF in l1["EF_eV"]]
l1["n_fermi_closed_cm2"] = [n_fermi_closed(EF, T, c) for EF in l1["EF_eV"]]
l1["correction_factor"] = l1["n_band_cm2"] / l1["n_fermi_cm2"]

assert np.all(np.diff(l1["correction_factor"]) > 0), "Correction factor is not monotonic."

EF_C2 = np.interp(2.0, l1["correction_factor"], l1["EF_eV"])
EF_C4 = np.interp(4.0, l1["correction_factor"], l1["EF_eV"])

# Exact operating point
n_band_op = n_band(EF_op, T, c)
n_fermi_op = n_fermi(EF_op, T, c)
correction_factor_op = n_band_op / n_fermi_op
mobility_corrected = 1.0 / (e * n_fermi_op * rho)
mobility_band = 1.0 / (e * n_band_op * rho)

print("\n=== Operating point ===")
print("EF_op =", EF_op, "eV")
print("n_band_op =", n_band_op, "cm^-2")
print("n_fermi_op =", n_fermi_op, "cm^-2")
print("correction factor =", correction_factor_op)
print("mobility_band =", mobility_band, "cm^2 V^-1 s^-1")
print("mobility_corrected =", mobility_corrected, "cm^2 V^-1 s^-1")
print("mu_corrected/mu_band =", mobility_corrected / mobility_band)
print("C=2 crossing =", EF_C2, "eV")
print("C=4 crossing =", EF_C4, "eV")

# Figure
plt.figure(figsize=(8, 5))

plt.plot(
    l1["EF_eV"],
    l1["correction_factor"],
    marker="o"
)

# C = 2 and C = 4 reference lines
plt.axhline(2.0, linestyle="--")
plt.axhline(4.0, linestyle="--")

plt.axvline(EF_C2, linestyle="--")
plt.axvline(EF_C4, linestyle="--")

# Exact operating point
plt.scatter(
    [EF_op],
    [correction_factor_op],
    s=80,
    label="G07 operating point"
)

# Numerical labels
plt.annotate(
    f"C=2 at {EF_C2:.4f} eV",
    (EF_C2, 2.0),
    xytext=(8, 10),
    textcoords="offset points"
)

plt.annotate(
    f"C=4 at {EF_C4:.4f} eV",
    (EF_C4, 4.0),
    xytext=(8, 10),
    textcoords="offset points"
)

plt.annotate(
    f"Operating point\nEF={EF_op:.4f} eV\nC={correction_factor_op:.3f}",
    (EF_op, correction_factor_op),
    xytext=(15, 15),
    textcoords="offset points"
)

plt.xlabel("Fermi level $E_F$ (eV)")
plt.ylabel("Correction factor $C=n_{band}/n_F$")
plt.title("L1 Carrier-count Correction Factor")
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    base / "L1_correction_factor.png",
    dpi=300
)

plt.show()

# results.json
pipeline_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True
).strip()

results["pipeline_commit"] = pipeline_commit
results["L1"]["correction_factor"] = float(correction_factor_op)
results["L1"]["n_fermi_op_cm2"] = float(n_fermi_op)
results["L1"]["mobility_corrected_cm2_Vs"] = float(mobility_corrected)

with open(base / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
