"""One-command entry point for the Group 7 analysis."""

from pathlib import Path

from l1_analysis import analyse


def main():
    root = Path(__file__).resolve().parent

    print("=== Reproducing G07 L1 ===")

    summary = analyse(
        data_dir=root,
        output_dir=root
    )

    print("\nG07 L1 reproduced successfully.")
    print(f"Operating EF: {summary['EF_op_eV']:.6f} eV")
    print(f"Correction factor: {summary['correction_factor']:.6f}")
    print(
        f"Corrected mobility: "
        f"{summary['mobility_corrected_cm2_Vs']:.6e} cm^2 V^-1 s^-1"
    )
    print(f"C=2 crossing: {summary['EF_at_C2_eV']:.6f} eV")
    print(f"C=4 crossing: {summary['EF_at_C4_eV']:.6f} eV")


if __name__ == "__main__":
    main()