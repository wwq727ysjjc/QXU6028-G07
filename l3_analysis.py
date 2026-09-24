from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent

A = np.load(ROOT / "L3_A.npy")
d_reps = np.loadtxt(ROOT / "L3_d.csv", delimiter=",")

# -------------------------
# 1. Basic checks
# -------------------------
assert A.shape == (128, 100)
assert d_reps.shape == (3, 128)

max_row_sum = np.max(np.abs(A.sum(axis=1)))
assert max_row_sum < 1e-11

dbar = d_reps.mean(axis=0)

print("A shape:", A.shape)
print("d repeats shape:", d_reps.shape)
print("max abs row sum:", max_row_sum)


# -------------------------
# 2. Gradient matrix
# -------------------------
def gradient_matrix(npix=10):
    rows = []
    N = npix * npix

    for i in range(npix):
        for j in range(npix):
            k = i * npix + j

            # Horizontal neighbour
            if j < npix - 1:
                r = np.zeros(N)
                r[k] = -1
                r[k + 1] = 1
                rows.append(r)

            # Vertical neighbour
            if i < npix - 1:
                r = np.zeros(N)
                r[k] = -1
                r[k + npix] = 1
                rows.append(r)

    return np.asarray(rows)


N = A.shape[1]
L = gradient_matrix(10)

assert L.shape == (180, 100)

print("L shape:", L.shape)


# -------------------------
# 3. Remove constant mode
# -------------------------
one = np.ones((N, 1)) / np.sqrt(N)
P = np.eye(N) - one @ one.T


# -------------------------
# 4. Reference regularisation
# -------------------------
eta0 = (
    1e-2
    * np.linalg.norm(A, 2) ** 2
    / np.linalg.norm(L, 2) ** 2
)

print("eta0:", eta0)


def reconstruct(dvec, eta=eta0):
    M = (
        P @ (A.T @ A + eta * (L.T @ L)) @ P
        + 1e-9 * np.eye(N)
    )

    b = P @ (A.T @ dvec)

    m = np.linalg.lstsq(M, b, rcond=None)[0]

    # Explicit zero-mean enforcement
    return m - m.mean()


# -------------------------
# 5. Reconstruct real data
# -------------------------
m = reconstruct(dbar)
mhat = m.reshape(10, 10)

print("map mean:", m.mean())
print("map min:", m.min())
print("map max:", m.max())
print("max |m|:", np.max(np.abs(m)))


# -------------------------
# 6. Feature coordinates
# -------------------------
coords = np.linspace(-2.7, 2.7, 10)

imax = np.argmax(mhat)
imin = np.argmin(mhat)

iy_max, ix_max = np.unravel_index(imax, mhat.shape)
iy_min, ix_min = np.unravel_index(imin, mhat.shape)

print(
    "largest positive feature:",
    mhat[iy_max, ix_max],
    "at (x,y) =",
    (coords[ix_max], coords[iy_max]),
)

print(
    "largest negative feature:",
    mhat[iy_min, ix_min],
    "at (x,y) =",
    (coords[ix_min], coords[iy_min]),
)


# -------------------------
# 7. Plot map
# -------------------------
fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(
    mhat,
    origin="lower",
    extent=[
        coords[0] - 0.3,
        coords[-1] + 0.3,
        coords[0] - 0.3,
        coords[-1] + 0.3,
    ],
    aspect="equal",
)

plt.colorbar(im, ax=ax, label="Relative non-uniformity")

ax.scatter(
    coords[ix_max],
    coords[iy_max],
    marker="x",
    s=100,
    label="Positive extremum",
)

ax.scatter(
    coords[ix_min],
    coords[iy_min],
    marker="+",
    s=100,
    label="Negative extremum",
)

ax.set_xlabel("x (probe pitches)")
ax.set_ylabel("y (probe pitches)")
ax.set_title("G07 L3 reconstructed zero-mean map")
ax.legend()

fig.tight_layout()
fig.savefig(ROOT / "L3_map.png", dpi=300)
plt.close(fig)

print("Saved: L3_map.png")


# ============================================================
# 8. Regularisation sensitivity
# ============================================================

eta_factors = np.array([0.01, 0.1, 1.0, 10.0, 100.0])

eta_values = eta0 * eta_factors
residual_norms = []
roughness_norms = []

for eta in eta_values:
    m_eta = reconstruct(dbar, eta=eta)

    residual = np.linalg.norm(A @ m_eta - dbar)
    roughness = np.linalg.norm(L @ m_eta)

    residual_norms.append(residual)
    roughness_norms.append(roughness)

    print(
        f"eta factor={eta/eta0:.2g}, "
        f"eta={eta:.6e}, "
        f"residual={residual:.6e}, "
        f"roughness={roughness:.6e}"
    )

# Use the prescribed reference eta unless there is a documented
# reason from noise/sensitivity to change it.
eta_chosen = eta0

print("chosen eta:", eta_chosen)


# ============================================================
# 9. Repeat-noise estimate
# ============================================================

sigma_j = np.std(d_reps, axis=0, ddof=1)

print("median channel sigma:", np.median(sigma_j))
print("mean channel sigma:", np.mean(sigma_j))
print("max channel sigma:", np.max(sigma_j))


# ============================================================
# 10. Resolution / point-spread test
# ============================================================

pixel_coords = np.linspace(-2.7, 2.7, 10)


def resolution_test(row, col):
    """
    Inject one positive pixel, enforce zero mean,
    forward-model it, reconstruct it, and estimate
    FWHM from positive-lobe weights.
    """

    q = np.zeros(N)
    k = row * 10 + col
    q[k] = 1.0

    # Required zero-mean constraint
    q = q - q.mean()

    y = A @ q

    qhat = reconstruct(y, eta=eta_chosen)
    qhat_map = qhat.reshape(10, 10)

    # Positive-lobe weighting
    w = np.maximum(qhat_map, 0.0)

    X, Y = np.meshgrid(pixel_coords, pixel_coords)

    x0 = pixel_coords[col]
    y0 = pixel_coords[row]

    r2 = (X - x0) ** 2 + (Y - y0) ** 2

    sigma2 = np.sum(w * r2) / (2.0 * np.sum(w))
    sigma = np.sqrt(sigma2)

    fwhm = 2.355 * sigma

    return qhat_map, fwhm


# One near-centre test and one off-centre test
psf_centre, fwhm_centre = resolution_test(4, 4)
psf_off, fwhm_off = resolution_test(2, 2)

resolution = max(fwhm_centre, fwhm_off)

print("central FWHM:", fwhm_centre, "probe pitches")
print("off-centre FWHM:", fwhm_off, "probe pitches")
print("conservative resolution:", resolution, "probe pitches")


# ============================================================
# 11. Noise-only detection limit
# ============================================================

NOISE_SEED = 6028
N_NOISE = 500

rng = np.random.default_rng(NOISE_SEED)

noise_maxima = []

for _ in range(N_NOISE):

    noise = rng.normal(
        loc=0.0,
        scale=sigma_j,
        size=len(sigma_j),
    )

    m_noise = reconstruct(noise, eta=eta_chosen)

    noise_maxima.append(
        np.max(np.abs(m_noise))
    )

noise_maxima = np.asarray(noise_maxima)

delta95 = np.percentile(noise_maxima, 95)

print("noise seed:", NOISE_SEED)
print("noise simulations:", N_NOISE)
print("delta95:", delta95)


# ============================================================
# 12. Specification decision
# ============================================================

threshold = 0.030
a = np.max(np.abs(m))

if a + delta95 <= threshold:
    decision = "pass"
    confidence = "clear pass"

elif a - delta95 > threshold:
    decision = "fail"
    confidence = "clear fail"

else:
    decision = "borderline"
    confidence = "borderline"

print("threshold:", threshold)
print("max |m|:", a)
print("max |m| + delta95:", a + delta95)
print("max |m| - delta95:", a - delta95)
print("decision:", decision)
print("confidence:", confidence)


# ============================================================
# 13. Method figure
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))


# Eta sensitivity
axes[0].plot(
    roughness_norms,
    residual_norms,
    marker="o",
)

for factor, xval, yval in zip(
    eta_factors,
    roughness_norms,
    residual_norms,
):
    axes[0].annotate(
        f"{factor:g}×",
        (xval, yval),
    )

axes[0].set_xlabel("Roughness norm ||Lm||")
axes[0].set_ylabel("Residual norm ||Am-d||")
axes[0].set_title("Regularisation sensitivity")


# Central point-spread
im1 = axes[1].imshow(
    psf_centre,
    origin="lower",
    extent=[-3, 3, -3, 3],
)

axes[1].set_title(
    f"Central PSF\nFWHM={fwhm_centre:.2f} pitches"
)
axes[1].set_xlabel("x (probe pitches)")
axes[1].set_ylabel("y (probe pitches)")
fig.colorbar(im1, ax=axes[1])


# Off-centre point-spread
im2 = axes[2].imshow(
    psf_off,
    origin="lower",
    extent=[-3, 3, -3, 3],
)

axes[2].set_title(
    f"Off-centre PSF\nFWHM={fwhm_off:.2f} pitches"
)
axes[2].set_xlabel("x (probe pitches)")
axes[2].set_ylabel("y (probe pitches)")
fig.colorbar(im2, ax=axes[2])

fig.tight_layout()

fig.savefig(
    ROOT / "L3_method.png",
    dpi=300,
)

plt.close(fig)

print("Saved: L3_method.png")


# ============================================================
# 14. Write L3 results into results.json
# ============================================================

results_path = ROOT / "results.json"

with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)

assert "L3" in results

results["L3"]["map"] = [float(x) for x in mhat.ravel(order="C")]

results["L3"]["features"] = [
    {
        "sign": "positive",
        "amplitude": float(mhat[iy_max, ix_max]),
        "x_probe_pitch": float(coords[ix_max]),
        "y_probe_pitch": float(coords[iy_max]),
    },
    {
        "sign": "negative",
        "amplitude": float(mhat[iy_min, ix_min]),
        "x_probe_pitch": float(coords[ix_min]),
        "y_probe_pitch": float(coords[iy_min]),
    },
]

results["L3"]["resolution_probe_pitch"] = float(resolution)
results["L3"]["detection_limit_fraction"] = float(delta95)

# JSON template expects a Boolean here
results["L3"]["pass"] = bool(decision == "pass")
results["L3"]["confidence"] = confidence

assert len(results["L3"]["map"]) == 100
assert np.all(np.isfinite(results["L3"]["map"]))

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Updated: results.json")
print("L3 map entries:", len(results["L3"]["map"]))