from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path(__file__).resolve().parent

DATA_FILE = BASE / "L2_data.csv"
META_FILE = BASE / "L2_meta.json"


# --------------------------------------------------
# Load raw data
# --------------------------------------------------
data = pd.read_csv(DATA_FILE)

with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

print("Group:", meta["group"])
print("Temperature:", meta["T_K"], "K")
print("Raw data shape:", data.shape)
print("Columns:", list(data.columns))


# --------------------------------------------------
# Basic checks
# --------------------------------------------------
required_columns = [
    "measurement_index",
    "repeat",
    "R_H_raw",
    "R_xx_raw",
]

assert list(data.columns) == required_columns
assert meta["group"] == "G07"

repeat_counts = data.groupby("measurement_index").size()

assert repeat_counts.nunique() == 1
assert repeat_counts.iloc[0] == 3

print("Unique measurement indices:", data["measurement_index"].nunique())
print("Repeats per index:", repeat_counts.iloc[0])


# --------------------------------------------------
# Average the three repeats
# Keep standard deviation for uncertainty later
# --------------------------------------------------
mean_data = (
    data.groupby("measurement_index", sort=True)
    .agg(
        R_H_mean=("R_H_raw", "mean"),
        R_H_std=("R_H_raw", "std"),
        R_xx_mean=("R_xx_raw", "mean"),
        R_xx_std=("R_xx_raw", "std"),
    )
    .reset_index()
)

print("\nMean-data shape:", mean_data.shape)


# --------------------------------------------------
# Locate Hall zero-crossing
# --------------------------------------------------
RH = mean_data["R_H_mean"].to_numpy()

crossings = []

for i in range(len(RH) - 1):
    if RH[i] == 0 or RH[i] * RH[i + 1] < 0:
        crossings.append(
            (
                int(mean_data.loc[i, "measurement_index"]),
                int(mean_data.loc[i + 1, "measurement_index"]),
            )
        )

print("\nR_H zero-crossing bracket(s):", crossings)

assert len(crossings) == 1

i_left, i_right = crossings[0]

row_left = mean_data.loc[
    mean_data["measurement_index"] == i_left
].iloc[0]

row_right = mean_data.loc[
    mean_data["measurement_index"] == i_right
].iloc[0]

print(
    f"Index {i_left}: "
    f"R_H = {row_left['R_H_mean']:.6e}, "
    f"R_xx = {row_left['R_xx_mean']:.6e}"
)

print(
    f"Index {i_right}: "
    f"R_H = {row_right['R_H_mean']:.6e}, "
    f"R_xx = {row_right['R_xx_mean']:.6e}"
)


# --------------------------------------------------
# Normalise the two observables independently
# IMPORTANT: preserve the sign of R_H
# --------------------------------------------------
RH_scale = np.max(np.abs(mean_data["R_H_mean"]))
Rxx_scale = np.max(mean_data["R_xx_mean"])

mean_data["R_H_norm"] = mean_data["R_H_mean"] / RH_scale
mean_data["R_xx_norm"] = mean_data["R_xx_mean"] / Rxx_scale

print("\nNormalisation scales:")
print(f"max |R_H| = {RH_scale:.6e}")
print(f"max R_xx   = {Rxx_scale:.6e}")

print("\nNormalised values around neutrality:")

print(
    mean_data.loc[
        mean_data["measurement_index"].isin([i_left, i_right]),
        [
            "measurement_index",
            "R_H_norm",
            "R_xx_norm",
        ],
    ]
)


# --------------------------------------------------
# Gate checks
# --------------------------------------------------
RH_norm = mean_data["R_H_norm"].to_numpy()

assert RH_norm[i_left] * RH_norm[i_right] < 0
assert np.isclose(np.max(np.abs(RH_norm)), 1.0)
assert np.isclose(np.max(mean_data["R_xx_norm"]), 1.0)

print("\nL2 GATE PASSED:")
print("1. Three repeats correctly grouped.")
print("2. Hall sign is preserved.")
print("3. Raw and normalised R_H cross zero at the same location.")
print("4. R_H and R_xx were normalised independently.")


# --------------------------------------------------
# Gate figure
# --------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Plot raw repeats
for repeat in sorted(data["repeat"].unique()):
    subset = data[data["repeat"] == repeat]

    axes[0].plot(
        subset["measurement_index"],
        subset["R_H_raw"],
        marker="o",
        markersize=3,
        linewidth=1,
        alpha=0.45,
        label=f"Repeat {repeat}",
    )

axes[0].plot(
    mean_data["measurement_index"],
    mean_data["R_H_mean"],
    linewidth=2,
    label="Mean",
)

axes[0].axhline(0, linewidth=1)
axes[0].axvspan(i_left, i_right, alpha=0.15)

axes[0].set_ylabel(r"$R_H$")
axes[0].set_title("L2 gate: raw Hall response")
axes[0].legend()


for repeat in sorted(data["repeat"].unique()):
    subset = data[data["repeat"] == repeat]

    axes[1].plot(
        subset["measurement_index"],
        subset["R_xx_raw"],
        marker="o",
        markersize=3,
        linewidth=1,
        alpha=0.45,
        label=f"Repeat {repeat}",
    )

axes[1].plot(
    mean_data["measurement_index"],
    mean_data["R_xx_mean"],
    linewidth=2,
    label="Mean",
)

axes[1].axvspan(i_left, i_right, alpha=0.15)

axes[1].set_xlabel("Measurement index")
axes[1].set_ylabel(r"$R_{xx}$")
axes[1].set_title("L2 gate: longitudinal response")

fig.tight_layout()

fig.savefig(
    BASE / "L2_gate_raw.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# Save processed mean data for checking
mean_data.to_csv(
    BASE / "L2_mean_normalised.csv",
    index=False,
)

print("\nSaved:")
print("L2_gate_raw.png")
print("L2_mean_normalised.csv")

# ==================================================
# L2 MODEL LIBRARY AND CENTRAL FIT
# ==================================================

kB = 8.617333e-5       # eV/K
NA = 3.816e15          # cm^-2
T = float(meta["T_K"])
kT = kB * T

# Required numerical grids from the Methods Guide
E_INT = np.linspace(-2.5, 2.5, 4000)
EF_GRID = np.linspace(-8.0 * kT, 9.0 * kT, 400)
GAP_GRID_KT = np.arange(0.50, 8.00 + 0.001, 0.25)

# The overall DoS scale cancels after independent normalisation.
C_DOS = 1.0


def fermi(E, EF, T):
    x = np.clip((E - EF) / (kB * T), -60, 60)
    return 1.0 / (np.exp(x) + 1.0)


def dos(E, model, c, Eg):
    """
    Density of states.

    cone:
        g(E) = c |E|

    hard_gap:
        g(E) = 0 inside |E| <= Eg/2
        g(E) = c (|E| - Eg/2) outside
    """
    if model == "cone":
        return c * np.abs(E)

    if model == "hard_gap":
        D = Eg / 2.0
        return np.where(
            np.abs(E) > D,
            c * (np.abs(E) - D),
            0.0,
        )

    raise ValueError(f"Unknown model: {model}")


def carrier_np(EF, T, model, c, Eg):
    """
    Calculate electron-like n and hole-like p populations
    using the Fermi-referenced split required for L2.
    """
    f = fermi(E_INT, EF, T)
    g = dos(E_INT, model, c, Eg)

    n = NA * np.trapezoid(
        np.where(E_INT >= EF, g * f, 0.0),
        E_INT,
    )

    p = NA * np.trapezoid(
        np.where(E_INT <= EF, g * (1.0 - f), 0.0),
        E_INT,
    )

    return n, p


def build_model_curve(model, Eg_kT=0.0):
    """
    Build the complete normalised R_H - R_xx model curve.

    For model curves we set e = 1 and mu = 1 because
    fixed scales cancel after independent normalisation.
    """
    Eg = Eg_kT * kT

    n_values = []
    p_values = []

    for EF in EF_GRID:
        n, p = carrier_np(
            EF=EF,
            T=T,
            model=model,
            c=C_DOS,
            Eg=Eg,
        )

        n_values.append(n)
        p_values.append(p)

    n_values = np.asarray(n_values)
    p_values = np.asarray(p_values)

    total = n_values + p_values

    # e = 1, mu = 1
    RH = (p_values - n_values) / (total ** 2)
    Rxx = 1.0 / total

    # Normalise each observable independently
    RH_norm = RH / np.max(np.abs(RH))
    Rxx_norm = Rxx / np.max(Rxx)

    return {
        "model": model,
        "Eg_kT": float(Eg_kT),
        "RH_norm": RH_norm,
        "Rxx_norm": Rxx_norm,
    }


def fit_curve_to_data(model_curve, data_xy):
    """
    For every measured pair, find the nearest point
    on the complete model curve.
    """
    model_xy = np.column_stack(
        [
            model_curve["RH_norm"],
            model_curve["Rxx_norm"],
        ]
    )

    d2 = (
        (
            data_xy[:, None, :]
            - model_xy[None, :, :]
        )
        ** 2
    ).sum(axis=2)

    j_nearest = np.argmin(d2, axis=1)

    rms = np.sqrt(
        np.mean(
            np.min(d2, axis=1)
        )
    )

    EF_assigned = EF_GRID[j_nearest]

    return {
        "rms": float(rms),
        "j_nearest": j_nearest,
        "EF_assigned": EF_assigned,
    }


# --------------------------------------------------
# Measured data in normalised pair space
# --------------------------------------------------
data_xy = mean_data[
    ["R_H_norm", "R_xx_norm"]
].to_numpy()


# --------------------------------------------------
# Fit ideal cone first
# --------------------------------------------------
cone_curve = build_model_curve(
    model="cone",
    Eg_kT=0.0,
)

cone_fit = fit_curve_to_data(
    cone_curve,
    data_xy,
)

r_cone = cone_fit["rms"]


# --------------------------------------------------
# Scan all hard-gap candidates
# --------------------------------------------------
gap_results = []
gap_library = {}

best_gap_curve = None
best_gap_fit = None
best_gap_kT = None

for Eg_kT in GAP_GRID_KT:

    curve = build_model_curve(
        model="hard_gap",
        Eg_kT=Eg_kT,
    )

    gap_library[float(Eg_kT)] = curve

    fit = fit_curve_to_data(
        curve,
        data_xy,
    )

    gap_results.append(
        {
            "Eg_kT": float(Eg_kT),
            "rms": fit["rms"],
        }
    )

    if (
        best_gap_fit is None
        or fit["rms"] < best_gap_fit["rms"]
    ):
        best_gap_fit = fit
        best_gap_curve = curve
        best_gap_kT = float(Eg_kT)


r_gap = best_gap_fit["rms"]


# --------------------------------------------------
# Official 15% model-selection rule
# --------------------------------------------------
gap_improves_enough = r_gap < 0.85 * r_cone
gap_is_resolvable = best_gap_kT >= 1.0

if gap_improves_enough and gap_is_resolvable:

    selected_model = "hard_gap"
    selected_Eg_kT = best_gap_kT
    selected_curve = best_gap_curve
    selected_fit = best_gap_fit

else:

    selected_model = "cone"
    selected_Eg_kT = 0.0
    selected_curve = cone_curve
    selected_fit = cone_fit


# --------------------------------------------------
# Print central-fit result
# --------------------------------------------------
print("\n==============================")
print("L2 CENTRAL MODEL FIT")
print("==============================")

print(f"Cone residual       = {r_cone:.8f}")
print(f"Best hard-gap Eg/kT = {best_gap_kT:.2f}")
print(f"Best gap residual   = {r_gap:.8f}")

print(
    "Gap/cone residual ratio = "
    f"{r_gap / r_cone:.6f}"
)

print(
    "15% improvement test: "
    f"{gap_improves_enough}"
)

print(
    "Eg >= 1 kBT test: "
    f"{gap_is_resolvable}"
)

print("\nOFFICIAL CENTRAL DECISION:")
print("Model =", selected_model)
print(f"Eg/kBT = {selected_Eg_kT:.2f}")

print(
    "Assigned EF points =",
    len(selected_fit["EF_assigned"]),
)

assert len(selected_fit["EF_assigned"]) == len(mean_data)


# --------------------------------------------------
# Save residual scan
# --------------------------------------------------
gap_df = pd.DataFrame(gap_results)

gap_df.to_csv(
    BASE / "L2_gap_residual_scan.csv",
    index=False,
)


# --------------------------------------------------
# Model figure: measured mean data + selected curve
# --------------------------------------------------
fig, ax = plt.subplots(
    figsize=(7.2, 6.2)
)

ax.plot(
    selected_curve["RH_norm"],
    selected_curve["Rxx_norm"],
    linewidth=2,
    label=(
        f"Selected model: {selected_model}, "
        f"Eg/kBT={selected_Eg_kT:.2f}"
    ),
)

ax.plot(
    mean_data["R_H_norm"],
    mean_data["R_xx_norm"],
    linewidth=1,
    alpha=0.5,
)

scatter = ax.scatter(
    mean_data["R_H_norm"],
    mean_data["R_xx_norm"],
    c=mean_data["measurement_index"],
    s=35,
    zorder=3,
)

ax.axvline(
    0,
    linewidth=1,
    linestyle="--",
)

ax.set_xlabel(r"Normalised $R_H$")
ax.set_ylabel(r"Normalised $R_{xx}$")

ax.set_title(
    "G07 L2: normalised transport pair and selected model"
)

ax.legend()

cbar = fig.colorbar(
    scatter,
    ax=ax,
)

cbar.set_label(
    "Measurement index"
)

fig.tight_layout()

fig.savefig(
    BASE / "L2_model_fit.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# --------------------------------------------------
# Residual-comparison figure
# --------------------------------------------------
fig, ax = plt.subplots(
    figsize=(7.2, 5.0)
)

ax.plot(
    gap_df["Eg_kT"],
    gap_df["rms"],
    marker="o",
    markersize=3,
    label="Hard-gap residual",
)

ax.axhline(
    r_cone,
    linestyle="--",
    label="Cone residual",
)

ax.axhline(
    0.85 * r_cone,
    linestyle=":",
    label="0.85 × cone residual",
)

ax.axvline(
    1.0,
    linestyle="--",
    label="1 kBT resolution threshold",
)

ax.scatter(
    [best_gap_kT],
    [r_gap],
    s=60,
    zorder=4,
    label="Best hard-gap fit",
)

ax.set_xlabel(r"$E_g/k_BT$")
ax.set_ylabel("RMS residual")

ax.set_title(
    "G07 L2: model residual comparison"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    BASE / "L2_residual_scan.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("\nSaved:")
print("L2_gap_residual_scan.csv")
print("L2_model_fit.png")
print("L2_residual_scan.png")


# ==================================================
# L2 BOOTSTRAP UNCERTAINTY
# ==================================================

N_BOOT = 200
BOOTSTRAP_SEED = 6028

rng = np.random.default_rng(BOOTSTRAP_SEED)

measurement_indices = np.sort(
    data["measurement_index"].unique()
)

n_index = len(measurement_indices)

bootstrap_Eg_kT = np.zeros(N_BOOT)
bootstrap_hard_gap = np.zeros(N_BOOT, dtype=bool)
bootstrap_EF = np.zeros((N_BOOT, n_index))


def normalise_pairs(RH, Rxx):
    """
    Independently normalise R_H and R_xx.
    Hall sign is preserved.
    """
    RH = np.asarray(RH, dtype=float)
    Rxx = np.asarray(Rxx, dtype=float)

    RH_norm = RH / np.max(np.abs(RH))
    Rxx_norm = Rxx / np.max(Rxx)

    return np.column_stack(
        [RH_norm, Rxx_norm]
    )


def select_model_for_pairs(data_xy):
    """
    Apply the exact same model-selection rule
    used for the central estimate.
    """

    # Cone fit
    cone_result = fit_curve_to_data(
        cone_curve,
        data_xy,
    )

    r_cone_boot = cone_result["rms"]

    # Best hard-gap fit
    best_gap_result = None
    best_gap_value = None
    best_gap_model_curve = None

    for Eg_kT in GAP_GRID_KT:

        Eg_key = float(Eg_kT)

        curve = gap_library[Eg_key]

        fit = fit_curve_to_data(
            curve,
            data_xy,
        )

        if (
            best_gap_result is None
            or fit["rms"] < best_gap_result["rms"]
        ):
            best_gap_result = fit
            best_gap_value = Eg_key
            best_gap_model_curve = curve

    r_gap_boot = best_gap_result["rms"]

    # Official 15% + 1 kBT rule
    choose_gap = (
        r_gap_boot < 0.85 * r_cone_boot
        and best_gap_value >= 1.0
    )

    if choose_gap:

        return {
            "model": "hard_gap",
            "Eg_kT": best_gap_value,
            "fit": best_gap_result,
            "curve": best_gap_model_curve,
            "r_cone": r_cone_boot,
            "r_gap": r_gap_boot,
        }

    else:

        return {
            "model": "cone",
            "Eg_kT": 0.0,
            "fit": cone_result,
            "curve": cone_curve,
            "r_cone": r_cone_boot,
            "r_gap": r_gap_boot,
        }


# --------------------------------------------------
# Prepare the three repeat rows for each index
# --------------------------------------------------
rows_by_index = {
    int(idx): (
        data[data["measurement_index"] == idx]
        .sort_values("repeat")
        .reset_index(drop=True)
    )
    for idx in measurement_indices
}


# --------------------------------------------------
# Run 200 bootstrap replicates
# --------------------------------------------------
for b in range(N_BOOT):

    RH_sample = []
    Rxx_sample = []

    for idx in measurement_indices:

        group = rows_by_index[int(idx)]

        # Randomly choose one of the three repeats
        j = rng.integers(0, len(group))

        sampled_row = group.iloc[j]

        RH_sample.append(
            sampled_row["R_H_raw"]
        )

        Rxx_sample.append(
            sampled_row["R_xx_raw"]
        )

    boot_xy = normalise_pairs(
        RH_sample,
        Rxx_sample,
    )

    boot_result = select_model_for_pairs(
        boot_xy
    )

    bootstrap_Eg_kT[b] = (
        boot_result["Eg_kT"]
    )

    bootstrap_hard_gap[b] = (
        boot_result["model"] == "hard_gap"
    )

    bootstrap_EF[b, :] = (
        boot_result["fit"]["EF_assigned"]
    )


# --------------------------------------------------
# Bootstrap summaries
# --------------------------------------------------
Eg_median = np.percentile(
    bootstrap_Eg_kT,
    50,
)

Eg_p16 = np.percentile(
    bootstrap_Eg_kT,
    16,
)

Eg_p84 = np.percentile(
    bootstrap_Eg_kT,
    84,
)

hard_gap_fraction = np.mean(
    bootstrap_hard_gap
)

EF_p16 = np.percentile(
    bootstrap_EF,
    16,
    axis=0,
)

EF_median = np.percentile(
    bootstrap_EF,
    50,
    axis=0,
)

EF_p84 = np.percentile(
    bootstrap_EF,
    84,
    axis=0,
)

central_EF = selected_fit["EF_assigned"]


# --------------------------------------------------
# Print bootstrap result
# --------------------------------------------------
print("\n==============================")
print("L2 BOOTSTRAP UNCERTAINTY")
print("==============================")

print(
    f"Bootstrap seed = {BOOTSTRAP_SEED}"
)

print(
    f"Replicates = {N_BOOT}"
)

print(
    "Eg/kBT bootstrap median = "
    f"{Eg_median:.3f}"
)

print(
    "Eg/kBT 16-84% interval = "
    f"[{Eg_p16:.3f}, {Eg_p84:.3f}]"
)

print(
    "Hard-gap selection fraction = "
    f"{hard_gap_fraction:.3f}"
)

print(
    "Hard-gap selections = "
    f"{bootstrap_hard_gap.sum()} / {N_BOOT}"
)

print(
    "Central model = "
    f"{selected_model}"
)

print(
    "Central Eg/kBT = "
    f"{selected_Eg_kT:.2f}"
)


# --------------------------------------------------
# Save trajectory table
# --------------------------------------------------
trajectory_df = pd.DataFrame(
    {
        "measurement_index": measurement_indices,
        "EF_central_eV": central_EF,
        "EF_bootstrap_median_eV": EF_median,
        "EF_p16_eV": EF_p16,
        "EF_p84_eV": EF_p84,
    }
)

trajectory_df.to_csv(
    BASE / "L2_EF_trajectory.csv",
    index=False,
)


# --------------------------------------------------
# Save bootstrap summary
# --------------------------------------------------
bootstrap_summary = {
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_replicates": N_BOOT,
    "central_model": selected_model,
    "central_Eg_kT": float(selected_Eg_kT),
    "Eg_bootstrap_median_kT": float(Eg_median),
    "Eg_bootstrap_p16_kT": float(Eg_p16),
    "Eg_bootstrap_p84_kT": float(Eg_p84),
    "hard_gap_selection_fraction": float(
        hard_gap_fraction
    ),
}

with open(
    BASE / "L2_bootstrap_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        bootstrap_summary,
        f,
        indent=2,
    )


# --------------------------------------------------
# Required trajectory figure
# --------------------------------------------------
fig, ax = plt.subplots(
    figsize=(8.5, 5.2)
)

ax.fill_between(
    measurement_indices,
    EF_p16,
    EF_p84,
    alpha=0.25,
    label="Bootstrap 16-84% interval",
)

ax.plot(
    measurement_indices,
    central_EF,
    marker="o",
    markersize=4,
    linewidth=1.8,
    label="Central assigned $E_F$",
)

ax.axhline(
    0,
    linewidth=1,
    linestyle="--",
)

ax.axvspan(
    i_left,
    i_right,
    alpha=0.12,
    label="Hall zero-crossing bracket",
)

ax.set_xlabel(
    "Measurement index"
)

ax.set_ylabel(
    r"Assigned $E_F$ (eV)"
)

ax.set_title(
    "G07 L2: inferred Fermi-level trajectory"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    BASE / "L2_EF_trajectory.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# --------------------------------------------------
# Basic trajectory checks
# --------------------------------------------------
assert len(central_EF) == n_index
assert len(EF_p16) == n_index
assert len(EF_p84) == n_index

assert np.all(
    EF_p16 <= EF_p84
)

assert np.all(
    np.isfinite(bootstrap_EF)
)

print("\nL2 BOOTSTRAP CHECKS PASSED")

print("\nSaved:")
print("L2_EF_trajectory.csv")
print("L2_bootstrap_summary.json")
print("L2_EF_trajectory.png")


# ==================================================
# WRITE L2 RESULTS TO results.json
# ==================================================

RESULTS_FILE = BASE / "results.json"

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

# Preserve L1 and L3; only update the L2 block
results["L2"]["model"] = selected_model
results["L2"]["Eg_kT"] = float(selected_Eg_kT)

# Bootstrap 16-84% interval is [0, 0] here,
# so the gap uncertainty is zero in kBT units.
results["L2"]["Eg_uncertainty_kT"] = float(
    max(
        selected_Eg_kT - Eg_p16,
        Eg_p84 - selected_Eg_kT,
    )
)

results["L2"]["EF_trajectory_eV"] = [
    float(x) for x in central_EF
]

# Required trajectory-length check
assert len(results["L2"]["EF_trajectory_eV"]) == n_index

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        indent=2,
    )

print("\n==============================")
print("L2 RESULTS.JSON UPDATED")
print("==============================")
print("model =", results["L2"]["model"])
print("Eg_kT =", results["L2"]["Eg_kT"])
print(
    "Eg_uncertainty_kT =",
    results["L2"]["Eg_uncertainty_kT"],
)
print(
    "EF trajectory length =",
    len(results["L2"]["EF_trajectory_eV"]),
)