"""
Tutorial: compare DAS and FBSS-MVBF on a real PyBF RF dataset.

This script turns the library-style beamformers into an executable learning
experiment:

    DataLoader -> RF tensor -> ImageSettings
        -> BFCartesianRealTime(DAS)
        -> BFMVBFspatial(FBSS-MVBF)
        -> side-by-side dB image comparison

Default output:
    tests/code/tutorial_mvbf_real_data/das_vs_mvbf_real_data.png

Run from the repository root:
    python tests/code/tutorial_mvbf_real_data/main.py

A faster, low-resolution run:
    python tests/code/tutorial_mvbf_real_data/main.py --image-res 40 60

A parameter experiment:
    python tests/code/tutorial_mvbf_real_data/main.py --channel-reduction 32 --window-width 8

A sweep over MVBF parameters:
    python tests/code/tutorial_mvbf_real_data/main.py --sweep \
        --sweep-channel-reductions 16 32 64 \
        --sweep-window-widths 4 8 12

A sweep over diagonal loading:
    python tests/code/tutorial_mvbf_real_data/main.py --loading-sweep \
        --channel-reduction 32 \
        --window-width 8 \
        --sweep-loading-scales 0.001 0.01 0.1 1.0

Profile metrics for one strong reflector:
    python tests/code/tutorial_mvbf_real_data/main.py --metrics --image-res 120 180

Circular CNR metrics, meaningful for contrast/cyst phantoms:
    python tests/code/tutorial_mvbf_real_data/main.py --cnr-metrics --image-res 120 180

Pixel-level MVBF weight diagnostics:
    python tests/code/tutorial_mvbf_real_data/main.py --weight-diagnostics --image-res 120 180

Apply receive apodization before MVBF covariance construction:
    python tests/code/tutorial_mvbf_real_data/main.py --mvbf-apply-apodization --metrics --image-res 120 180
"""

from pathlib import Path
import argparse
import csv
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# File location:
#   <repo>/tests/code/tutorial_mvbf_real_data/main.py
# Import root required by this project layout:
#   <repo_parent>
PATH_TO_LIB = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PATH_TO_LIB))

from pybf.pybf.io_interfaces import DataLoader
from pybf.pybf.image_settings import ImageSettings
from pybf.scripts.beamformer_cartesian_realtime import BFCartesianRealTime
from pybf.scripts.beamformer_mvbf_spatial_smooth import BFMVBFspatial


DEFAULT_DATASET = PATH_TO_LIB / "pybf" / "tests" / "data" / "rf_dataset.hdf5"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "das_vs_mvbf_real_data.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PyBF DAS and FBSS-MVBF on a real RF dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to rf_dataset.hdf5. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path of the saved comparison figure. Default: %(default)s",
    )
    parser.add_argument(
        "--image-res",
        type=int,
        nargs=2,
        default=[40, 60],
        metavar=("NX", "NZ"),
        help="Image resolution as lateral and axial pixel counts. Default: %(default)s",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=[-0.022, 0.022],
        metavar=("X0", "X1"),
        help="Lateral image range in meters. Default: %(default)s",
    )
    parser.add_argument(
        "--z-range",
        type=float,
        nargs=2,
        default=[0.0014784, 0.040],
        metavar=("Z0", "Z1"),
        help="Axial image range in meters. Default: %(default)s",
    )
    parser.add_argument(
        "--decimation-factor",
        type=int,
        default=1,
        help="RF preprocessing decimation factor. Default: %(default)s",
    )
    parser.add_argument(
        "--interpolation-factor",
        type=int,
        default=4,
        help="RF/IQ interpolation factor. Default: %(default)s",
    )
    parser.add_argument(
        "--channel-reduction",
        type=int,
        default=32,
        help="Centered receive aperture size used by both DAS apodization and MVBF. Default: %(default)s",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=8,
        help="MVBF spatial smoothing subarray length L. Default: %(default)s",
    )
    parser.add_argument(
        "--diagonal-loading-scale",
        type=float,
        default=1.0,
        help="Scale factor for MVBF diagonal loading. 1.0 reproduces the original PyBF behavior. Default: %(default)s",
    )
    parser.add_argument(
        "--mvbf-apply-apodization",
        action="store_true",
        help=(
            "Apply receive apodization weights to MVBF input channels before covariance construction. "
            "Default off reproduces the original PyBF behavior."
        ),
    )
    parser.add_argument(
        "--num-acqs",
        type=int,
        default=None,
        help="Number of LRIs / plane waves to use from the frame. Default: use all available.",
    )
    parser.add_argument(
        "--db-range",
        type=float,
        default=50.0,
        help="Displayed dynamic range in dB. Default: %(default)s",
    )
    parser.add_argument(
        "--no-das-warmup",
        action="store_true",
        help="Skip the first DAS warmup run. By default DAS is warmed up to exclude Numba compile time.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="After DAS/MVBF reconstruction, measure lateral profile metrics around an automatically selected bright reflector.",
    )
    parser.add_argument(
        "--cnr-metrics",
        action="store_true",
        help="After DAS/MVBF reconstruction, measure circular contrast-to-noise ratio. This is meaningful for contrast/cyst phantoms.",
    )
    parser.add_argument(
        "--weight-diagnostics",
        action="store_true",
        help="Print pixel-level MVBF covariance/weight diagnostics for selected x-z points.",
    )
    parser.add_argument(
        "--diagnostic-points",
        type=float,
        nargs="+",
        default=[
            0.000, 0.02235,
            -0.006, 0.02235,
            0.000, 0.03450,
            -0.010, 0.03450,
        ],
        help="Diagnostic points as repeated x z values in meters. Default samples bright and nearby background locations.",
    )
    parser.add_argument(
        "--cnr-circles",
        type=float,
        nargs="+",
        default=[
            -0.00043, 0.01492, 0.0035, 0.00172,
            -0.00043, 0.04279, 0.0035, 0.00172,
            -0.00720, 0.02829, 0.0063, 0.00315,
        ],
        help=(
            "Circular CNR ROIs as repeated x z outer_radius inner_radius values in meters. "
            "Defaults are the PICMUS contrast circles used by the original PyBF example."
        ),
    )
    parser.add_argument(
        "--metric-roi",
        type=float,
        nargs=4,
        default=[-0.006, 0.006, 0.018, 0.028],
        metavar=("X0", "X1", "Z0", "Z1"),
        help="ROI in meters used to auto-pick a bright reflector for profile metrics. Default: %(default)s",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a grid sweep over channel_reduction and window_width values.",
    )
    parser.add_argument(
        "--sweep-channel-reductions",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Receive aperture sizes to test in sweep mode. Default: %(default)s",
    )
    parser.add_argument(
        "--sweep-window-widths",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        help="MVBF subarray lengths L to test in sweep mode. Default: %(default)s",
    )
    parser.add_argument(
        "--loading-sweep",
        action="store_true",
        help="Sweep diagonal_loading_scale while keeping channel_reduction and window_width fixed.",
    )
    parser.add_argument(
        "--sweep-loading-scales",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0],
        help="Diagonal loading scales to test in loading-sweep mode. Default: %(default)s",
    )
    return parser.parse_args()


def load_rf_data(data_loader, num_acqs=None):
    available = data_loader.num_of_acq_per_frame
    if num_acqs is None:
        num_acqs = available
    if num_acqs < 1 or num_acqs > available:
        raise ValueError("num_acqs must be between 1 and {}".format(available))

    rf_data = np.stack(
        [data_loader.get_rf_data(0, acq_idx) for acq_idx in range(num_acqs)],
        axis=0,
    )
    return rf_data


def subset_tx_strategy(tx_strategy, num_acqs):
    """Keep tx_strategy consistent when only the first num_acqs LRIs are used."""
    strategy_name = tx_strategy[0]
    strategy_params = tx_strategy[1]
    return [strategy_name, strategy_params[:num_acqs]]


def build_beamformers(args, data_loader):
    trans = data_loader.transducer
    x0, x1 = args.x_range
    z0, z1 = args.z_range

    img_config = ImageSettings(x0, x1, z0, z1, 5, trans)
    num_acqs = data_loader.num_of_acq_per_frame if args.num_acqs is None else args.num_acqs
    tx_strategy = subset_tx_strategy(data_loader.tx_strategy, num_acqs)

    common = dict(
        f_sampling=data_loader.f_sampling,
        tx_strategy=tx_strategy,
        transducer_obj=trans,
        decimation_factor=args.decimation_factor,
        interpolation_factor=args.interpolation_factor,
        image_res=args.image_res,
        img_config_obj=img_config,
        start_time=data_loader.hardware.start_time,
        correction_time_shift=data_loader.hardware.correction_time_shift,
        alpha_fov_apod=40,
        bp_filter_params=None,
        envelope_detector="I_Q",
        picmus_dataset=data_loader.simulation_flag,
        channel_reduction=args.channel_reduction,
    )

    das = BFCartesianRealTime(**common)
    mvbf = BFMVBFspatial(
        **common,
        window_width=args.window_width,
        diagonal_loading_scale=args.diagonal_loading_scale,
        apply_apodization=args.mvbf_apply_apodization,
    )
    return das, mvbf


def beamform_and_time(label, func):
    print("\nRunning {}...".format(label))
    start = time.time()
    image = func()
    elapsed = time.time() - start
    print("{} wall time: {:.4f} s".format(label, elapsed))
    return image, elapsed


def to_db(image, db_range):
    magnitude = np.abs(image).astype(np.float64)
    magnitude = magnitude / (np.max(magnitude) + 1e-12)
    image_db = 20.0 * np.log10(magnitude + 1e-12)
    return np.clip(image_db, -db_range, 0.0)


def save_comparison(args, das_db, mvbf_db, das_time, mvbf_time):
    diff_db = mvbf_db - das_db
    x0, x1 = args.x_range
    z0, z1 = args.z_range
    extent = [x0, x1, z1, z0]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    im0 = axes[0].imshow(
        das_db,
        cmap="gray",
        extent=extent,
        aspect="auto",
        vmin=-args.db_range,
        vmax=0,
    )
    axes[0].set_title("DAS, dB")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("z [m]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        mvbf_db,
        cmap="gray",
        extent=extent,
        aspect="auto",
        vmin=-args.db_range,
        vmax=0,
    )
    axes[1].set_title("FBSS-MVBF, dB")
    axes[1].set_xlabel("x [m]")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    diff_limit = min(12.0, args.db_range)
    im2 = axes[2].imshow(
        diff_db,
        cmap="seismic",
        extent=extent,
        aspect="auto",
        vmin=-diff_limit,
        vmax=diff_limit,
    )
    axes[2].set_title("MVBF - DAS, dB")
    axes[2].set_xlabel("x [m]")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(
        "PyBF real RF data: DAS vs FBSS-MVBF, image_res={}, aperture={}, L={}\n"
        "DAS {:.3f}s, MVBF {:.3f}s".format(
            args.image_res,
            args.channel_reduction,
            args.window_width,
            das_time,
            mvbf_time,
        )
    )
    fig.savefig(str(args.output), dpi=160)
    plt.close(fig)

    return diff_db


def image_axes(args):
    nx, nz = args.image_res
    x0, x1 = args.x_range
    z0, z1 = args.z_range
    return np.linspace(x0, x1, nx), np.linspace(z0, z1, nz)


def first_crossing_width(axis, profile_db, level_db):
    """Return width around the peak at a relative dB level."""
    peak_index = int(np.argmax(profile_db))

    left = peak_index
    while left > 0 and profile_db[left] >= level_db:
        left -= 1

    right = peak_index
    while right < profile_db.shape[0] - 1 and profile_db[right] >= level_db:
        right += 1

    if left == peak_index or right == peak_index:
        return np.nan

    def interp_crossing(i0, i1):
        x0, x1 = axis[i0], axis[i1]
        y0, y1 = profile_db[i0], profile_db[i1]
        if abs(y1 - y0) < 1e-12:
            return x1
        return x0 + (level_db - y0) * (x1 - x0) / (y1 - y0)

    left_x = interp_crossing(left, left + 1)
    right_x = interp_crossing(right, right - 1)
    return abs(right_x - left_x)


def peak_sidelobe_level(profile_db, exclusion_width_pixels=5):
    """Crude PSL estimate outside a small main-lobe exclusion window."""
    peak_index = int(np.argmax(profile_db))
    mask = np.ones(profile_db.shape[0], dtype=bool)
    lo = max(0, peak_index - exclusion_width_pixels)
    hi = min(profile_db.shape[0], peak_index + exclusion_width_pixels + 1)
    mask[lo:hi] = False
    if not np.any(mask):
        return np.nan
    return float(np.max(profile_db[mask]))


def profile_metrics(axis_x, profile_abs):
    profile_abs = profile_abs.astype(np.float64)
    profile_norm = profile_abs / (np.max(profile_abs) + 1e-12)
    profile_db = 20.0 * np.log10(profile_norm + 1e-12)
    return {
        "profile_db": profile_db,
        "fwhm_m": first_crossing_width(axis_x, profile_db, -6.0),
        "width_20db_m": first_crossing_width(axis_x, profile_db, -20.0),
        "psl_db": peak_sidelobe_level(profile_db),
    }


def save_profile_metrics(args, das_image, mvbf_image):
    axis_x, axis_z = image_axes(args)
    das_abs = np.abs(das_image)
    mvbf_abs = np.abs(mvbf_image)

    x0, x1, z0, z1 = args.metric_roi
    x_mask = (axis_x >= x0) & (axis_x <= x1)
    z_mask = (axis_z >= z0) & (axis_z <= z1)
    if not np.any(x_mask) or not np.any(z_mask):
        raise ValueError("metric ROI does not overlap the image axes")

    roi = das_abs[np.ix_(z_mask, x_mask)]
    local_z, local_x = np.unravel_index(np.argmax(roi), roi.shape)
    z_indices = np.where(z_mask)[0]
    x_indices = np.where(x_mask)[0]
    peak_z_index = int(z_indices[local_z])
    peak_x_index = int(x_indices[local_x])

    das_profile = das_abs[peak_z_index, :]
    mvbf_profile = mvbf_abs[peak_z_index, :]
    das_metrics = profile_metrics(axis_x, das_profile)
    mvbf_metrics = profile_metrics(axis_x, mvbf_profile)

    stem = args.output.with_suffix("")
    profile_png = stem.parent / (stem.name + "_profile_metrics.png")
    profile_csv = stem.parent / (stem.name + "_profile_metrics.csv")
    profile_png.parent.mkdir(parents=True, exist_ok=True)

    with profile_csv.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=[
            "method",
            "peak_x_m",
            "peak_z_m",
            "fwhm_mm",
            "width_20db_mm",
            "psl_db",
        ])
        writer.writeheader()
        for method, metrics in [("DAS", das_metrics), ("MVBF", mvbf_metrics)]:
            writer.writerow({
                "method": method,
                "peak_x_m": "{:.8f}".format(axis_x[peak_x_index]),
                "peak_z_m": "{:.8f}".format(axis_z[peak_z_index]),
                "fwhm_mm": "{:.6f}".format(metrics["fwhm_m"] * 1e3),
                "width_20db_mm": "{:.6f}".format(metrics["width_20db_m"] * 1e3),
                "psl_db": "{:.6f}".format(metrics["psl_db"]),
            })

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(axis_x * 1e3, das_metrics["profile_db"], label="DAS")
    ax.plot(axis_x * 1e3, mvbf_metrics["profile_db"], label="FBSS-MVBF")
    ax.axhline(-6, color="gray", linestyle="--", linewidth=1, label="-6 dB")
    ax.axhline(-20, color="gray", linestyle=":", linewidth=1, label="-20 dB")
    ax.axvline(axis_x[peak_x_index] * 1e3, color="k", linestyle=":", linewidth=1)
    ax.set_ylim([-50, 2])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Normalized lateral profile [dB]")
    ax.set_title(
        "Lateral profile at z={:.2f} mm, auto peak x={:.2f} mm".format(
            axis_z[peak_z_index] * 1e3,
            axis_x[peak_x_index] * 1e3,
        )
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(str(profile_png), dpi=160)
    plt.close(fig)

    print("\nProfile metrics around auto-selected DAS bright reflector:")
    print("  ROI [x0, x1, z0, z1] m: {}".format(args.metric_roi))
    print("  selected peak: x={:.3f} mm, z={:.3f} mm".format(
        axis_x[peak_x_index] * 1e3,
        axis_z[peak_z_index] * 1e3,
    ))
    print("  Method | FWHM(mm) | width@-20dB(mm) | crude PSL(dB)")
    for method, metrics in [("DAS", das_metrics), ("MVBF", mvbf_metrics)]:
        print("  {:5s} | {:8.3f} | {:15.3f} | {:12.2f}".format(
            method,
            metrics["fwhm_m"] * 1e3,
            metrics["width_20db_m"] * 1e3,
            metrics["psl_db"],
        ))
    print("  CSV: {}".format(profile_csv))
    print("  Profile figure: {}".format(profile_png))

    return {
        "peak_x_m": float(axis_x[peak_x_index]),
        "peak_z_m": float(axis_z[peak_z_index]),
        "das": das_metrics,
        "mvbf": mvbf_metrics,
        "csv": profile_csv,
        "figure": profile_png,
    }


def parse_cnr_circles(flat_values):
    values = np.asarray(flat_values, dtype=np.float64)
    if values.size % 4 != 0:
        raise ValueError("--cnr-circles must contain groups of 4 values: x z outer_radius inner_radius")
    circles = values.reshape(-1, 4)
    for circle in circles:
        if circle[3] <= 0 or circle[2] <= 0 or circle[3] >= circle[2]:
            raise ValueError("Each CNR circle must satisfy 0 < inner_radius < outer_radius")
    return circles


def cnr_for_circle(image, axis_x, axis_z, circle):
    center_x, center_z, outer_radius, inner_radius = circle
    grid_x, grid_z = np.meshgrid(axis_x, axis_z)
    radius = np.sqrt((grid_x - center_x) ** 2 + (grid_z - center_z) ** 2)
    inner_mask = radius < inner_radius
    outer_mask = (radius >= inner_radius) & (radius < outer_radius)

    magnitude = np.abs(image).astype(np.float64)
    inner = magnitude[inner_mask]
    outer = magnitude[outer_mask]

    if inner.size < 2 or outer.size < 2:
        return {
            "valid": False,
            "cnr_db": np.nan,
            "inner_mean": np.nan,
            "outer_mean": np.nan,
            "inner_std": np.nan,
            "outer_std": np.nan,
            "inner_pixels": int(inner.size),
            "outer_pixels": int(outer.size),
        }

    inner_mean = float(np.mean(inner))
    outer_mean = float(np.mean(outer))
    inner_var = float(np.var(inner))
    outer_var = float(np.var(outer))
    denominator = 0.5 * np.sqrt(inner_var + outer_var)
    cnr_db = 20.0 * np.log10((abs(outer_mean - inner_mean) + 1e-12) / (denominator + 1e-12))

    return {
        "valid": True,
        "cnr_db": float(cnr_db),
        "inner_mean": inner_mean,
        "outer_mean": outer_mean,
        "inner_std": float(np.sqrt(inner_var)),
        "outer_std": float(np.sqrt(outer_var)),
        "inner_pixels": int(inner.size),
        "outer_pixels": int(outer.size),
    }


def save_cnr_metrics(args, das_image, mvbf_image):
    axis_x, axis_z = image_axes(args)
    circles = parse_cnr_circles(args.cnr_circles)

    stem = args.output.with_suffix("")
    cnr_csv = stem.parent / (stem.name + "_cnr_metrics.csv")
    cnr_png = stem.parent / (stem.name + "_cnr_metrics.png")
    cnr_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for circle_index, circle in enumerate(circles, start=1):
        for method, image in [("DAS", das_image), ("MVBF", mvbf_image)]:
            metrics = cnr_for_circle(image, axis_x, axis_z, circle)
            row = {
                "circle": circle_index,
                "method": method,
                "x_m": circle[0],
                "z_m": circle[1],
                "outer_radius_m": circle[2],
                "inner_radius_m": circle[3],
                **metrics,
            }
            rows.append(row)

    with cnr_csv.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=[
            "circle",
            "method",
            "x_m",
            "z_m",
            "outer_radius_m",
            "inner_radius_m",
            "valid",
            "cnr_db",
            "inner_mean",
            "outer_mean",
            "inner_std",
            "outer_std",
            "inner_pixels",
            "outer_pixels",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    das_db = to_db(das_image, args.db_range)
    mvbf_db = to_db(mvbf_image, args.db_range)
    x0, x1 = args.x_range
    z0, z1 = args.z_range
    extent = [x0, x1, z1, z0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, image_db, title in [
        (axes[0], das_db, "DAS with CNR ROIs"),
        (axes[1], mvbf_db, "MVBF with CNR ROIs"),
    ]:
        im = ax.imshow(
            image_db,
            cmap="gray",
            extent=extent,
            aspect="auto",
            vmin=-args.db_range,
            vmax=0,
        )
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        for circle in circles:
            center_x, center_z, outer_radius, inner_radius = circle
            ax.add_patch(Circle((center_x, center_z), outer_radius, edgecolor="red", fill=False, lw=1.5))
            ax.add_patch(Circle((center_x, center_z), inner_radius, edgecolor="lime", fill=False, lw=1.5))

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046)
    fig.suptitle("Circular CNR ROIs: red=outer annulus boundary, green=inner target")
    fig.savefig(str(cnr_png), dpi=160)
    plt.close(fig)

    print("\nCircular CNR metrics:")
    print("  Note: these default circles are meaningful for PICMUS contrast/cyst data.")
    print("        On the bundled point/line-scatterer dataset, treat them as a dry run only.")
    print("  Circle | Method | CNR(dB) | inner mean | outer mean | inner px | outer px")
    for row in rows:
        print("  {:6d} | {:6s} | {:7.2f} | {:10.3g} | {:10.3g} | {:8d} | {:8d}".format(
            int(row["circle"]),
            row["method"],
            row["cnr_db"],
            row["inner_mean"],
            row["outer_mean"],
            int(row["inner_pixels"]),
            int(row["outer_pixels"]),
        ))
    print("  CSV: {}".format(cnr_csv))
    print("  ROI figure: {}".format(cnr_png))

    return {
        "rows": rows,
        "csv": cnr_csv,
        "figure": cnr_png,
    }


def parse_diagnostic_points(flat_values):
    values = np.asarray(flat_values, dtype=np.float64)
    if values.size % 2 != 0:
        raise ValueError("--diagnostic-points must contain repeated x z pairs")
    return values.reshape(-1, 2)


def nearest_image_point(args, x_m, z_m):
    axis_x, axis_z = image_axes(args)
    ix = int(np.argmin(np.abs(axis_x - x_m)))
    iz = int(np.argmin(np.abs(axis_z - z_m)))
    point_index = iz * args.image_res[0] + ix
    return ix, iz, point_index, axis_x[ix], axis_z[iz]


def mvbf_pixel_core_from_vector(channel_vector, subarray_length, loading_scale):
    channel_vector = np.asarray(channel_vector, dtype=np.complex128)
    aperture_size = channel_vector.shape[0]
    n_snapshots = aperture_size - subarray_length + 1
    snapshots = np.zeros((n_snapshots, subarray_length), dtype=np.complex128)

    for snapshot_index in range(n_snapshots):
        snapshots[snapshot_index] = channel_vector[snapshot_index:snapshot_index + subarray_length]

    r_forward = np.zeros((subarray_length, subarray_length), dtype=np.complex128)
    r_backward = np.zeros((subarray_length, subarray_length), dtype=np.complex128)
    for snapshot in snapshots:
        r_forward += np.outer(np.conj(snapshot), snapshot)
        reversed_snapshot = np.flip(snapshot)
        r_backward += np.outer(np.conj(reversed_snapshot), reversed_snapshot)

    loading = loading_scale * np.identity(subarray_length) * np.trace(r_forward) / subarray_length
    r_loaded = 0.5 * r_forward + 0.5 * r_backward + loading
    r_inv = np.linalg.inv(r_loaded)
    steering = np.ones(subarray_length, dtype=np.complex128)
    numerator = np.matmul(r_inv, steering)
    weights = numerator / np.sum(numerator)

    # This mirrors scripts/beamformer_mvbf_spatial_smooth.py exactly:
    # x_sum = np.sum(corr_array, axis=0)
    # result = 1/K * np.sum(w_tilda.T * x_sum)
    x_sum = np.sum(snapshots, axis=0)
    pybf_mvbf_output = np.sum(weights.T * x_sum) / n_snapshots

    # A conventional complex inner-product convention would often use
    # conj(weights). This is not used by PyBF here, but is useful to diagnose
    # whether the complex convention matters at the selected pixel.
    conjugate_weight_output = np.sum(np.conj(weights).T * x_sum) / n_snapshots

    return {
        "snapshots": snapshots,
        "r_forward": r_forward,
        "r_backward": r_backward,
        "r_loaded": r_loaded,
        "weights": weights,
        "pybf_mvbf_output": pybf_mvbf_output,
        "conjugate_weight_output": conjugate_weight_output,
        "condition_number": float(np.linalg.cond(r_loaded)),
        "weight_l1": float(np.sum(np.abs(weights))),
        "weight_l2": float(np.linalg.norm(weights)),
        "weight_max_abs": float(np.max(np.abs(weights))),
        "weight_sum": np.sum(weights),
        "n_snapshots": int(n_snapshots),
    }


def save_weight_diagnostics(args, mvbf, rf_data, das_image, mvbf_image):
    points = parse_diagnostic_points(args.diagnostic_points)
    n_elements = int(mvbf._transducer.num_of_elements)
    channel_reduction = mvbf.channel_reduction
    subarray_length = mvbf.window_width
    start_i = int(np.ceil((n_elements - channel_reduction) / 2))
    stop_i = int(start_i + channel_reduction)

    stem = args.output.with_suffix("")
    csv_path = stem.parent / (stem.name + "_weight_diagnostics.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print("\nMVBF pixel-level weight diagnostics:")
    print("  channel_reduction={}, L={}, diagonal_loading_scale={}".format(
        channel_reduction,
        subarray_length,
        mvbf.diagonal_loading_scale,
    ))
    print("  MVBF apply_apodization={}".format(mvbf.apply_apodization))
    print("  Important source-code observation:")
    print("    DAS multiplies samples by full apodization weights.")
    if mvbf.apply_apodization:
        print("    MVBF now also applies apodization before covariance construction.")
    else:
        print("    Original MVBF only uses apodization as a zero/nonzero mask, then forms weights from un-apodized samples.")

    for point_id, (x_req, z_req) in enumerate(points, start=1):
        ix, iz, point_index, x_actual, z_actual = nearest_image_point(args, x_req, z_req)
        das_pixel = das_image[iz, ix]
        mvbf_pixel = mvbf_image[iz, ix]
        apod_selected = mvbf._apod[point_index, start_i:stop_i]
        apod_active = int(np.count_nonzero(apod_selected))

        print("\n" + "-" * 78)
        print("Point {} requested x={:.3f} mm, z={:.3f} mm".format(
            point_id, x_req * 1e3, z_req * 1e3
        ))
        print("  nearest pixel ix={}, iz={}, x={:.3f} mm, z={:.3f} mm".format(
            ix, iz, x_actual * 1e3, z_actual * 1e3
        ))
        print("  reconstructed |DAS|={:.4g}, |MVBF|={:.4g}, MVBF/DAS={:.4g}".format(
            np.abs(das_pixel),
            np.abs(mvbf_pixel),
            np.abs(mvbf_pixel) / (np.abs(das_pixel) + 1e-12),
        ))
        print("  selected aperture active apod elements: {}/{}".format(apod_active, channel_reduction))
        print("  apod sum={:.4g}, apod L2={:.4g}, apod max={:.4g}".format(
            np.sum(apod_selected),
            np.linalg.norm(apod_selected),
            np.max(apod_selected) if apod_selected.size else 0.0,
        ))

        for acq_index in range(rf_data.shape[0]):
            rf_data_proc = mvbf._preprocess_data(rf_data[acq_index, :, :])
            rf_data_proc_trans = np.transpose(rf_data_proc)
            delays_samples = mvbf._rx_delays_samples + mvbf._tx_delays_samples[acq_index, :]
            sample_indices = delays_samples[:, point_index].astype(np.int64)

            padded = np.zeros((rf_data_proc_trans.shape[0] + 1, rf_data_proc_trans.shape[1]), dtype=np.complex128)
            padded[:rf_data_proc_trans.shape[0], :] = rf_data_proc_trans
            sample_indices = sample_indices.copy()
            sample_indices[sample_indices >= rf_data_proc_trans.shape[0] - 1] = -1

            channel_indices = np.arange(n_elements)
            aligned_vector = padded[sample_indices, channel_indices][start_i:stop_i]
            aligned_vector_apod = aligned_vector * apod_selected

            if apod_active == 0 or channel_reduction < subarray_length:
                note = "skipped"
                row = {
                    "point": point_id,
                    "acq": acq_index,
                    "x_m": x_actual,
                    "z_m": z_actual,
                    "active_apod": apod_active,
                    "aligned_l2": np.linalg.norm(aligned_vector),
                    "apod_aligned_l2": np.linalg.norm(aligned_vector_apod),
                    "das_sum_abs": np.abs(np.sum(aligned_vector_apod)),
                    "mvbf_abs": np.nan,
                    "mvbf_apod_abs": np.nan,
                    "conj_mvbf_abs": np.nan,
                    "cond": np.nan,
                    "weight_l1": np.nan,
                    "weight_l2": np.nan,
                    "weight_max_abs": np.nan,
                    "note": note,
                }
                rows.append(row)
                continue

            core = mvbf_pixel_core_from_vector(
                aligned_vector,
                subarray_length,
                mvbf.diagonal_loading_scale,
            )
            core_apod = mvbf_pixel_core_from_vector(
                aligned_vector_apod,
                subarray_length,
                mvbf.diagonal_loading_scale,
            )

            das_sum = np.sum(aligned_vector_apod)
            das_sum_unapod = np.sum(aligned_vector)
            mvbf_out = core["pybf_mvbf_output"]
            mvbf_apod_out = core_apod["pybf_mvbf_output"]
            conj_out = core["conjugate_weight_output"]

            row = {
                "point": point_id,
                "acq": acq_index,
                "x_m": x_actual,
                "z_m": z_actual,
                "active_apod": apod_active,
                "aligned_l2": np.linalg.norm(aligned_vector),
                "apod_aligned_l2": np.linalg.norm(aligned_vector_apod),
                "das_sum_abs": np.abs(das_sum),
                "das_unapod_sum_abs": np.abs(das_sum_unapod),
                "mvbf_abs": np.abs(mvbf_out),
                "mvbf_apod_abs": np.abs(mvbf_apod_out),
                "conj_mvbf_abs": np.abs(conj_out),
                "cond": core["condition_number"],
                "weight_l1": core["weight_l1"],
                "weight_l2": core["weight_l2"],
                "weight_max_abs": core["weight_max_abs"],
                "weight_sum_real": np.real(core["weight_sum"]),
                "weight_sum_imag": np.imag(core["weight_sum"]),
                "note": "",
            }
            rows.append(row)

            print("  acq {}: |aligned|_2={:.4g}, |apod aligned|_2={:.4g}".format(
                acq_index, row["aligned_l2"], row["apod_aligned_l2"]
            ))
            print("         |DAS apod sum|={:.4g}, |DAS un-apod sum|={:.4g}".format(
                row["das_sum_abs"], row["das_unapod_sum_abs"]
            ))
            print("         |MVBF PyBF|={:.4g}, |MVBF if apodized input|={:.4g}, |MVBF conj(w)|={:.4g}".format(
                row["mvbf_abs"], row["mvbf_apod_abs"], row["conj_mvbf_abs"]
            ))
            print("         cond(R)={:.3g}, ||w||1={:.3g}, ||w||2={:.3g}, max|w|={:.3g}, sum(w)={:.3g}{:+.3g}j".format(
                row["cond"],
                row["weight_l1"],
                row["weight_l2"],
                row["weight_max_abs"],
                row["weight_sum_real"],
                row["weight_sum_imag"],
            ))

    if rows:
        fieldnames = sorted(rows[0].keys())
        with csv_path.open("w", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print("\nWeight diagnostics CSV:")
        print("  {}".format(csv_path))

    return rows


def save_sweep_grid(args, sweep_results, output_path):
    valid_results = [result for result in sweep_results if result["mvbf_db"] is not None]
    if not valid_results:
        return

    channel_values = sorted(set(result["channel_reduction"] for result in valid_results))
    window_values = sorted(set(result["window_width"] for result in valid_results))
    n_rows = len(channel_values)
    n_cols = len(window_values)

    x0, x1 = args.x_range
    z0, z1 = args.z_range
    extent = [x0, x1, z1, z0]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    by_key = {
        (result["channel_reduction"], result["window_width"]): result
        for result in valid_results
    }

    for row, channel_reduction in enumerate(channel_values):
        for col, window_width in enumerate(window_values):
            ax = axes[row][col]
            result = by_key.get((channel_reduction, window_width))
            if result is None:
                ax.axis("off")
                continue

            im = ax.imshow(
                result["mvbf_db"],
                cmap="gray",
                extent=extent,
                aspect="auto",
                vmin=-args.db_range,
                vmax=0,
            )
            ax.set_title(
                "aperture={}, L={}\n{:.2f}s, 95%|diff|={:.1f} dB".format(
                    channel_reduction,
                    window_width,
                    result["time"],
                    result["diff_abs_p95"],
                )
            )
            ax.set_xlabel("x [m]")
            if col == 0:
                ax.set_ylabel("z [m]")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018)
    fig.suptitle("FBSS-MVBF parameter sweep")
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def write_sweep_csv(sweep_results, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "channel_reduction",
        "window_width",
        "valid",
        "time_s",
        "mean_db",
        "std_db",
        "diff_mean_db",
        "diff_std_db",
        "diff_abs_p95_db",
        "note",
    ]
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for result in sweep_results:
            writer.writerow({
                "channel_reduction": result["channel_reduction"],
                "window_width": result["window_width"],
                "valid": result["valid"],
                "time_s": "{:.6f}".format(result["time"]) if result["time"] is not None else "",
                "mean_db": "{:.6f}".format(result["mean_db"]) if result["mean_db"] is not None else "",
                "std_db": "{:.6f}".format(result["std_db"]) if result["std_db"] is not None else "",
                "diff_mean_db": "{:.6f}".format(result["diff_mean"]) if result["diff_mean"] is not None else "",
                "diff_std_db": "{:.6f}".format(result["diff_std"]) if result["diff_std"] is not None else "",
                "diff_abs_p95_db": "{:.6f}".format(result["diff_abs_p95"]) if result["diff_abs_p95"] is not None else "",
                "note": result["note"],
            })


def run_sweep(args, data_loader, rf_data):
    print("\nSweep mode enabled.")
    print("  channel_reductions: {}".format(args.sweep_channel_reductions))
    print("  window_widths: {}".format(args.sweep_window_widths))

    # Build a DAS reference once. For a fair visual comparison, use the largest
    # aperture requested in this sweep as the DAS reference aperture.
    reference_args = argparse.Namespace(**vars(args))
    reference_args.channel_reduction = max(args.sweep_channel_reductions)
    reference_args.window_width = min(args.sweep_window_widths)

    das_reference, _ = build_beamformers(reference_args, data_loader)
    if not args.no_das_warmup:
        print("\nWarming up DAS reference once.")
        das_reference.beamform(rf_data, numba_active=True)

    das_image, das_time = beamform_and_time(
        "DAS reference / aperture {}".format(reference_args.channel_reduction),
        lambda: das_reference.beamform(rf_data, numba_active=True),
    )
    das_db = to_db(das_image, args.db_range)

    sweep_results = []

    for channel_reduction in args.sweep_channel_reductions:
        for window_width in args.sweep_window_widths:
            print("\n" + "=" * 78)
            print("Sweep case: channel_reduction={}, window_width={}".format(
                channel_reduction, window_width
            ))

            result = {
                "channel_reduction": channel_reduction,
                "window_width": window_width,
                "valid": False,
                "time": None,
                "mean_db": None,
                "std_db": None,
                "diff_mean": None,
                "diff_std": None,
                "diff_abs_p95": None,
                "mvbf_db": None,
                "note": "",
            }

            if window_width > channel_reduction:
                result["note"] = "Skipped: window_width must be <= channel_reduction."
                print(result["note"])
                sweep_results.append(result)
                continue

            case_args = argparse.Namespace(**vars(args))
            case_args.channel_reduction = channel_reduction
            case_args.window_width = window_width

            try:
                _, mvbf = build_beamformers(case_args, data_loader)
                mvbf_image, mvbf_time = beamform_and_time(
                    "FBSS-MVBF / aperture {}, L {}".format(channel_reduction, window_width),
                    lambda: mvbf.beamform(rf_data, numba_active=False),
                )
                mvbf_db = to_db(mvbf_image, args.db_range)
                diff_db = mvbf_db - das_db

                result.update({
                    "valid": True,
                    "time": mvbf_time,
                    "mean_db": float(mvbf_db.mean()),
                    "std_db": float(mvbf_db.std()),
                    "diff_mean": float(diff_db.mean()),
                    "diff_std": float(diff_db.std()),
                    "diff_abs_p95": float(np.percentile(np.abs(diff_db), 95)),
                    "mvbf_db": mvbf_db,
                })
                print("  mean/std dB: {:.2f} / {:.2f}".format(
                    result["mean_db"], result["std_db"]
                ))
                print("  diff to DAS mean/std/95%abs: {:.2f} / {:.2f} / {:.2f} dB".format(
                    result["diff_mean"], result["diff_std"], result["diff_abs_p95"]
                ))

            except Exception as exc:
                result["note"] = "Failed: {}".format(exc)
                print(result["note"])

            sweep_results.append(result)

    stem = args.output.with_suffix("")
    csv_path = stem.parent / (stem.name + "_sweep.csv")
    grid_path = stem.parent / (stem.name + "_sweep_grid.png")
    write_sweep_csv(sweep_results, csv_path)
    save_sweep_grid(args, sweep_results, grid_path)

    print("\nSweep summary:")
    print("  DAS reference aperture: {}".format(reference_args.channel_reduction))
    print("  DAS reference time: {:.4f} s".format(das_time))
    print("  CSV: {}".format(csv_path))
    print("  Grid figure: {}".format(grid_path))

    compact = [
        result for result in sweep_results
        if result["valid"]
    ]
    if compact:
        print("\nCompact table:")
        print("aperture | L | MVBF time(s) | MVBF mean(dB) | 95% |MV-DAS|(dB)")
        for result in compact:
            print("{:8d} | {:2d} | {:12.4f} | {:13.2f} | {:16.2f}".format(
                result["channel_reduction"],
                result["window_width"],
                result["time"],
                result["mean_db"],
                result["diff_abs_p95"],
            ))

    return sweep_results


def save_loading_sweep_grid(args, loading_results, output_path):
    valid_results = [result for result in loading_results if result["valid"]]
    if not valid_results:
        return

    n_cols = len(valid_results)
    x0, x1 = args.x_range
    z0, z1 = args.z_range
    extent = [x0, x1, z1, z0]

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(4.0 * n_cols, 4.0),
        squeeze=False,
        constrained_layout=True,
    )

    for col, result in enumerate(valid_results):
        ax = axes[0][col]
        im = ax.imshow(
            result["mvbf_db"],
            cmap="gray",
            extent=extent,
            aspect="auto",
            vmin=-args.db_range,
            vmax=0,
        )
        ax.set_title(
            "loading={}\n{:.2f}s, mean={:.1f} dB".format(
                result["loading_scale"],
                result["time"],
                result["mean_db"],
            )
        )
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel("z [m]")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.028)
    fig.suptitle(
        "FBSS-MVBF diagonal loading sweep, aperture={}, L={}".format(
            args.channel_reduction,
            args.window_width,
        )
    )
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def run_loading_sweep(args, data_loader, rf_data):
    print("\nLoading sweep mode enabled.")
    print("  channel_reduction: {}".format(args.channel_reduction))
    print("  window_width: {}".format(args.window_width))
    print("  loading scales: {}".format(args.sweep_loading_scales))

    das, _ = build_beamformers(args, data_loader)
    if not args.no_das_warmup:
        print("\nWarming up DAS reference once.")
        das.beamform(rf_data, numba_active=True)
    das_image, das_time = beamform_and_time(
        "DAS reference / aperture {}".format(args.channel_reduction),
        lambda: das.beamform(rf_data, numba_active=True),
    )
    das_db = to_db(das_image, args.db_range)

    loading_results = []
    for loading_scale in args.sweep_loading_scales:
        print("\n" + "=" * 78)
        print("Loading case: diagonal_loading_scale={}".format(loading_scale))

        case_args = argparse.Namespace(**vars(args))
        case_args.diagonal_loading_scale = loading_scale

        result = {
            "loading_scale": loading_scale,
            "valid": False,
            "time": None,
            "mean_db": None,
            "std_db": None,
            "diff_mean": None,
            "diff_std": None,
            "diff_abs_p95": None,
            "mvbf_db": None,
            "note": "",
        }

        try:
            _, mvbf = build_beamformers(case_args, data_loader)
            mvbf_image, mvbf_time = beamform_and_time(
                "FBSS-MVBF / loading {}".format(loading_scale),
                lambda: mvbf.beamform(rf_data, numba_active=False),
            )
            mvbf_db = to_db(mvbf_image, args.db_range)
            diff_db = mvbf_db - das_db

            result.update({
                "valid": True,
                "time": mvbf_time,
                "mean_db": float(mvbf_db.mean()),
                "std_db": float(mvbf_db.std()),
                "diff_mean": float(diff_db.mean()),
                "diff_std": float(diff_db.std()),
                "diff_abs_p95": float(np.percentile(np.abs(diff_db), 95)),
                "mvbf_db": mvbf_db,
            })
            print("  mean/std dB: {:.2f} / {:.2f}".format(
                result["mean_db"], result["std_db"]
            ))
            print("  diff to DAS mean/std/95%abs: {:.2f} / {:.2f} / {:.2f} dB".format(
                result["diff_mean"], result["diff_std"], result["diff_abs_p95"]
            ))

        except Exception as exc:
            result["note"] = "Failed: {}".format(exc)
            print(result["note"])

        loading_results.append(result)

    stem = args.output.with_suffix("")
    csv_path = stem.parent / (stem.name + "_loading_sweep.csv")
    grid_path = stem.parent / (stem.name + "_loading_sweep_grid.png")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=[
            "loading_scale",
            "valid",
            "time_s",
            "mean_db",
            "std_db",
            "diff_mean_db",
            "diff_std_db",
            "diff_abs_p95_db",
            "note",
        ])
        writer.writeheader()
        for result in loading_results:
            writer.writerow({
                "loading_scale": result["loading_scale"],
                "valid": result["valid"],
                "time_s": "{:.6f}".format(result["time"]) if result["time"] is not None else "",
                "mean_db": "{:.6f}".format(result["mean_db"]) if result["mean_db"] is not None else "",
                "std_db": "{:.6f}".format(result["std_db"]) if result["std_db"] is not None else "",
                "diff_mean_db": "{:.6f}".format(result["diff_mean"]) if result["diff_mean"] is not None else "",
                "diff_std_db": "{:.6f}".format(result["diff_std"]) if result["diff_std"] is not None else "",
                "diff_abs_p95_db": "{:.6f}".format(result["diff_abs_p95"]) if result["diff_abs_p95"] is not None else "",
                "note": result["note"],
            })

    save_loading_sweep_grid(args, loading_results, grid_path)

    print("\nLoading sweep summary:")
    print("  DAS reference time: {:.4f} s".format(das_time))
    print("  CSV: {}".format(csv_path))
    print("  Grid figure: {}".format(grid_path))

    print("\nCompact table:")
    print("loading | MVBF time(s) | MVBF mean(dB) | 95% |MV-DAS|(dB)")
    for result in loading_results:
        if result["valid"]:
            print("{:7g} | {:12.4f} | {:13.2f} | {:16.2f}".format(
                result["loading_scale"],
                result["time"],
                result["mean_db"],
                result["diff_abs_p95"],
            ))

    return loading_results


def main():
    args = parse_args()

    print("PyBF tutorial: DAS vs FBSS-MVBF on real RF data")
    print("Script: {}".format(Path(__file__).resolve()))
    print("Dataset: {}".format(args.dataset))
    print("Output: {}".format(args.output))
    print("Image resolution: {}".format(args.image_res))
    print("Channel reduction: {}".format(args.channel_reduction))
    print("MVBF window width L: {}".format(args.window_width))

    data_loader = DataLoader(str(args.dataset))
    try:
        print("\nDataset metadata:")
        print("  frames: {}".format(data_loader.num_of_frames))
        print("  LRIs per frame: {}".format(data_loader.num_of_acq_per_frame))
        print("  tx_strategy: {}".format(data_loader.tx_strategy))

        rf_data = load_rf_data(data_loader, args.num_acqs)
        print("  loaded rf_data shape: {}".format(rf_data.shape))
        print("  meaning: (acquisitions, elements, samples)")

        if args.sweep:
            run_sweep(args, data_loader, rf_data)
            return

        if args.loading_sweep:
            run_loading_sweep(args, data_loader, rf_data)
            return

        das, mvbf = build_beamformers(args, data_loader)

        if not args.no_das_warmup:
            print("\nWarming up DAS Numba path once, so the reported DAS time excludes JIT compilation.")
            das.beamform(rf_data, numba_active=True)

        das_image, das_time = beamform_and_time(
            "DAS / BFCartesianRealTime",
            lambda: das.beamform(rf_data, numba_active=True),
        )
        mvbf_image, mvbf_time = beamform_and_time(
            "FBSS-MVBF / BFMVBFspatial",
            lambda: mvbf.beamform(rf_data, numba_active=False),
        )

    finally:
        data_loader.close_file()

    das_db = to_db(das_image, args.db_range)
    mvbf_db = to_db(mvbf_image, args.db_range)
    diff_db = save_comparison(args, das_db, mvbf_db, das_time, mvbf_time)

    if args.metrics:
        save_profile_metrics(args, das_image, mvbf_image)

    if args.cnr_metrics:
        save_cnr_metrics(args, das_image, mvbf_image)

    if args.weight_diagnostics:
        save_weight_diagnostics(args, mvbf, rf_data, das_image, mvbf_image)

    print("\nImage statistics:")
    print("  DAS dB min/max/mean: {:.2f} / {:.2f} / {:.2f}".format(
        das_db.min(), das_db.max(), das_db.mean()
    ))
    print("  MVBF dB min/max/mean: {:.2f} / {:.2f} / {:.2f}".format(
        mvbf_db.min(), mvbf_db.max(), mvbf_db.mean()
    ))
    print("  MVBF-DAS dB diff mean/std/95%abs: {:.2f} / {:.2f} / {:.2f}".format(
        diff_db.mean(), diff_db.std(), np.percentile(np.abs(diff_db), 95)
    ))
    print("\nSaved comparison figure:")
    print("  {}".format(args.output))


if __name__ == "__main__":
    main()
