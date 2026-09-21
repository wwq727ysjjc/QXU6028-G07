"""Reproducible Group 7 L1 carrier-count analysis.

Reads only the supplied regular_G07 L1 files, validates the numerical model,
writes the required L1 JSON fields, and generates the correction-factor figure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KB_EV_K = 8.617333e-5
N_A_CM2 = 3.816e15
ELEMENTARY_CHARGE_C = 1.602176634e-19
E_MAX_EV = 2.5
N_INTEGRATION_POINTS = 20_001


def fermi(E_eV: np.ndarray, EF_eV: float, T_K: float) -> np.ndarray:
    """Finite-temperature Fermi-Dirac occupation (energies in eV)."""
    x = np.clip((E_eV - EF_eV) / (KB_EV_K * T_K), -60.0, 60.0)
    return 1.0 / (np.exp(x) + 1.0)


def n_band_cm2(EF_eV: float, T_K: float, c_per_eV2: float) -> float:
    """Band-edge count: N_A integral_0^Emax c E f(E) dE."""
    E = np.linspace(0.0, E_MAX_EV, N_INTEGRATION_POINTS)
    integrand = c_per_eV2 * E * fermi(E, EF_eV, T_K)
    return float(N_A_CM2 * np.trapezoid(integrand, E))


def n_fermi_cm2(EF_eV: float, T_K: float, c_per_eV2: float) -> float:
    """Fermi-referenced count: N_A integral_EF^Emax c E f(E) dE."""
    E = np.linspace(EF_eV, E_MAX_EV, N_INTEGRATION_POINTS)
    integrand = c_per_eV2 * E * fermi(E, EF_eV, T_K)
    return float(N_A_CM2 * np.trapezoid(integrand, E))


def n_fermi_closed_cm2(EF_eV: float, T_K: float, c_per_eV2: float) -> float:
    """Closed form for the ideal linear-DoS Fermi-referenced integral."""
    kT_eV = KB_EV_K * T_K
    return float(
        N_A_CM2
        * c_per_eV2
        * kT_eV
        * (EF_eV * np.log(2.0) + (np.pi**2 / 12.0) * kT_eV)
    )


def correction_factor(EF_eV: float, T_K: float, c_per_eV2: float) -> float:
    return n_band_cm2(EF_eV, T_K, c_per_eV2) / n_fermi_cm2(
        EF_eV, T_K, c_per_eV2
    )


def crossing_eV(
    target: float,
    T_K: float,
    c_per_eV2: float,
    lo_eV: float = 0.0,
    hi_eV: float = 0.30,
    tolerance_eV: float = 1e-10,
) -> float:
    """Find C(E_F)=target by bisection on the continuous integral model."""
    f_lo = correction_factor(lo_eV, T_K, c_per_eV2) - target
    f_hi = correction_factor(hi_eV, T_K, c_per_eV2) - target
    if f_lo * f_hi > 0:
        raise ValueError(f"Target C={target:g} is not bracketed by [{lo_eV}, {hi_eV}] eV")
    while hi_eV - lo_eV > tolerance_eV:
        mid_eV = (lo_eV + hi_eV) / 2.0
        f_mid = correction_factor(mid_eV, T_K, c_per_eV2) - target
        if f_lo * f_mid <= 0:
            hi_eV = mid_eV
        else:
            lo_eV = mid_eV
            f_lo = f_mid
    return (lo_eV + hi_eV) / 2.0

def get_git_commit(repo_dir: Path) -> str:
    """Return the Git commit used to generate the submitted outputs."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        text=True
    ).strip()


def analyse(data_dir: Path, output_dir: Path) -> dict:
    l1_path = data_dir / "L1_data.csv"
    params_path = data_dir / "L1_params.json"
    template_path = data_dir / "results_template.json"
    for path in (l1_path, params_path, template_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Group 7 file not found: {path}")

    l1 = pd.read_csv(l1_path)
    required_columns = {"EF_eV", "n_band_cm2", "rho_ohm_sq"}
    if not required_columns.issubset(l1.columns):
        raise ValueError(f"L1_data.csv must contain {sorted(required_columns)}")

    params = json.loads(params_path.read_text(encoding="utf-8"))
    if params.get("group") != "G07" or params.get("dataset_set") != "regular":
        raise ValueError("Refusing to run: inputs are not the regular Group 7 dataset")

    c = float(params["c"])
    T = float(params["T"])
    EF_op = float(params["EF_op_eV"])
    rho = float(params["rho_ohm_sq"])

    # The Methods Guide requires the supplied n_band column for sweep data.
    l1["n_fermi_cm2"] = [n_fermi_cm2(x, T, c) for x in l1["EF_eV"]]
    l1["n_fermi_closed_cm2"] = [
        n_fermi_closed_cm2(x, T, c) for x in l1["EF_eV"]
    ]
    l1["correction_factor"] = l1["n_band_cm2"] / l1["n_fermi_cm2"]

    # Verification 1: numerical and closed-form n_F must agree.
    closed_rel_error = np.max(
        np.abs(l1["n_fermi_cm2"] - l1["n_fermi_closed_cm2"])
        / l1["n_fermi_closed_cm2"]
    )
    # A 1e-6 threshold is stringent relative to the requested precision while
    # allowing the finite 2.5 eV cutoff and 20,001-point trapezoidal grid.
    if closed_rel_error > 1e-6:
        raise AssertionError(f"Closed-form check failed: max relative error={closed_rel_error:.3e}")

    # Verification 2: supplied band counts must match recomputation closely.
    l1["n_band_recomputed_cm2"] = [n_band_cm2(x, T, c) for x in l1["EF_eV"]]
    supplied_band_rel_error = np.max(
        np.abs(l1["n_band_cm2"] - l1["n_band_recomputed_cm2"])
        / l1["n_band_cm2"]
    )
    if supplied_band_rel_error > 2e-8:
        raise AssertionError(
            f"Supplied n_band check failed: max relative error={supplied_band_rel_error:.3e}"
        )

    if not np.all(np.diff(l1["correction_factor"]) > 0):
        raise AssertionError("Correction factor must increase over the positive-EF sweep")

    gate_EF = np.array([0.0, 0.073, 0.100, 0.200])
    gate_expected = np.array([1.00, 2.00, 2.60, 5.10])
    gate_calculated = np.array([correction_factor(x, T, c) for x in gate_EF])
    if not np.allclose(gate_calculated, gate_expected, atol=0.04, rtol=0.0):
        raise AssertionError("One or more Methods Guide correction-factor gates failed")

    EF_C2 = crossing_eV(2.0, T, c)
    EF_C4 = crossing_eV(4.0, T, c)

    # Recompute both counts at the exact JSON operating point, not a CSV row.
    n_band_op = n_band_cm2(EF_op, T, c)
    n_fermi_op = n_fermi_cm2(EF_op, T, c)
    C_op = n_band_op / n_fermi_op
    mobility_corrected = 1.0 / (ELEMENTARY_CHARGE_C * n_fermi_op * rho)
    mobility_band = 1.0 / (ELEMENTARY_CHARGE_C * n_band_op * rho)

    output_dir.mkdir(parents=True, exist_ok=True)
    l1.to_csv(output_dir / "L1_sweep_calculated.csv", index=False)
    pd.DataFrame(
        {
            "EF_eV": gate_EF,
            "expected_C_approx": gate_expected,
            "calculated_C": gate_calculated,
        }
    ).to_csv(output_dir / "L1_validation_gates.csv", index=False)

    results = json.loads(template_path.read_text(encoding="utf-8"))

pipeline_commit = get_git_commit(data_dir)

results["pipeline_commit"] = pipeline_commit

results["L1"] = {
    "correction_factor": C_op,
    "n_fermi_op_cm2": n_fermi_op,
    "mobility_corrected_cm2_Vs": mobility_corrected,
}

    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    (output_dir / "commit_hash.txt").write_text(
    pipeline_commit + "\n",
    encoding="utf-8"
)

    summary = {
        "group": "G07",
        "dataset_set": "regular",
        "EF_op_eV": EF_op,
        "rho_ohm_sq": rho,
        "n_band_op_cm2": n_band_op,
        "n_fermi_op_cm2": n_fermi_op,
        "correction_factor": C_op,
        "mobility_band_cm2_Vs": mobility_band,
        "mobility_corrected_cm2_Vs": mobility_corrected,
        "mobility_ratio_corrected_to_band": mobility_corrected / mobility_band,
        "EF_at_C2_eV": EF_C2,
        "EF_at_C4_eV": EF_C4,
        "max_closed_form_relative_error": float(closed_rel_error),
        "max_supplied_band_relative_error": float(supplied_band_rel_error),
        "gate_values": dict(zip([str(x) for x in gate_EF], gate_calculated.tolist())),
    }
    (output_dir / "L1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    # Supplied sweep values are shown as data markers; the continuous integral model
    # is used for the smooth curve, exact operating point, and root-found crossings.
    EF_dense = np.linspace(0.0, max(0.22, float(l1["EF_eV"].max())), 350)
    C_dense = np.array([correction_factor(x, T, c) for x in EF_dense])
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(EF_dense, C_dense, color="#1f77b4", linewidth=2.0, label="Integral model")
    ax.scatter(
        l1["EF_eV"],
        l1["correction_factor"],
        s=30,
        facecolor="white",
        edgecolor="#1f77b4",
        zorder=3,
        label="Supplied G07 sweep",
    )
    for target, crossing, color in [(2.0, EF_C2, "#d95f02"), (4.0, EF_C4, "#7570b3")]:
        ax.axhline(target, color=color, linestyle="--", linewidth=1.1)
        ax.axvline(crossing, color=color, linestyle=":", linewidth=1.2)
        ax.annotate(
            f"C={target:g} at $E_F$={crossing:.4f} eV",
            xy=(crossing, target),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )
    ax.scatter(
        [EF_op],
        [C_op],
        s=95,
        marker="*",
        color="#e41a1c",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
        label=f"Operating point: {EF_op:.4f} eV, C={C_op:.3f}",
    )
    ax.set_xlabel(r"Fermi level $E_F$ (eV)")
    ax.set_ylabel(r"Correction factor $C=n_{band}/n_F$")
    ax.set_title("Group 7 L1 carrier-count correction")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.8)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "L1_correction_factor.png", dpi=300)
    fig.savefig(output_dir / "L1_correction_factor.pdf")
    plt.close(fig)
    return summary




def main() -> None:
    base = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base)
    parser.add_argument("--output-dir", type=Path, default=base)

    args = parser.parse_args()

    summary = analyse(args.data_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()