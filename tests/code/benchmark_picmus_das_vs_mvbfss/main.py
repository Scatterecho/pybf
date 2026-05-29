"""
Benchmark DAS versus FBSS-MVBF on the PICMUS contrast/speckle dataset.

This script is intended for fair engineering comparisons:
    - use the same receive aperture for DAS and MVBFss
    - optionally apply receive apodization before MVBF covariance estimation
    - report runtime and circular CNR metrics
    - save a side-by-side dB comparison figure and CSV metrics

Default run from repository root:
    python tests/code/benchmark_picmus_das_vs_mvbfss/main.py

Examples:
    python tests/code/benchmark_picmus_das_vs_mvbfss/main.py --image-res 80 120 --channel-reduction 128
    python tests/code/benchmark_picmus_das_vs_mvbfss/main.py --channel-reduction 64 --window-width 8
    python tests/code/benchmark_picmus_das_vs_mvbfss/main.py --no-mvbf-apply-apodization
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


PATH_TO_LIB = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PATH_TO_LIB))

from pybf.pybf.io_interfaces import DataLoader
from pybf.pybf.image_settings import ImageSettings
from pybf.scripts.beamformer_DAS_ref import BFCartesianReference
from pybf.scripts.beamformer_mvbf_spatial_smooth import BFMVBFspatial


DEFAULT_DATASET = PATH_TO_LIB / "pybf" / "tests" / "data" / "Picmus" / "contrast_speckle" / "rf_dataset.hdf5"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


DEFAULT_CNR_CIRCLES = np.asarray([
    [-0.00043, 0.01492, 0.0035, 0.00172],
    [-0.00043, 0.04279, 0.0035, 0.00172],
    [-0.00720, 0.02829, 0.0063, 0.00315],
], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(description="PICMUS DAS vs FBSS-MVBF benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-res", type=int, nargs=2, default=[80, 120], metavar=("NX", "NZ"))
    parser.add_argument("--x-range", type=float, nargs=2, default=[-0.019, 0.019], metavar=("X0", "X1"))
    parser.add_argument("--z-range", type=float, nargs=2, default=[0.005, 0.050], metavar=("Z0", "Z1"))
    parser.add_argument("--channel-reduction", type=int, default=128, help="Receive aperture for both DAS and MVBFss")
    parser.add_argument("--window-width", type=int, default=8, help="MVBFss spatial smoothing subarray length L")
    parser.add_argument("--interpolation-factor", type=int, default=10)
    parser.add_argument("--decimation-factor", type=int, default=1)
    parser.add_argument("--alpha-fov-apod", type=float, default=40)
    parser.add_argument("--db-range", type=float, default=50)
    parser.add_argument("--plane-wave-indices", type=int, nargs="+", default=[33, 37, 38, 42])
    parser.add_argument("--mvbf-apply-apodization", dest="mvbf_apply_apodization", action="store_true", default=True)
    parser.add_argument("--no-mvbf-apply-apodization", dest="mvbf_apply_apodization", action="store_false")
    return parser.parse_args()


def load_rf_stack(data_loader, plane_wave_indices):
    rf_data = np.stack([
        data_loader.get_rf_data(0, int(index))
        for index in plane_wave_indices
    ], axis=0)
    tx_angles = [data_loader.tx_strategy[1][int(index)] for index in plane_wave_indices]
    max_angle_deg = int(round(np.max(np.abs(np.asarray(tx_angles))) * 180.0 / np.pi))
    tx_strategy = [
        "PW_{}_{}".format(len(plane_wave_indices), max_angle_deg),
        tx_angles,
    ]
    return rf_data, tx_strategy


def build_beamformers(args, data_loader, tx_strategy):
    x0, x1 = args.x_range
    z0, z1 = args.z_range
    img_config = ImageSettings(x0, x1, z0, z1, 5, data_loader.transducer)

    common = dict(
        f_sampling=data_loader.f_sampling,
        tx_strategy=tx_strategy,
        transducer_obj=data_loader.transducer,
        decimation_factor=args.decimation_factor,
        interpolation_factor=args.interpolation_factor,
        image_res=args.image_res,
        img_config_obj=img_config,
        start_time=0,
        correction_time_shift=0,
        alpha_fov_apod=args.alpha_fov_apod,
        bp_filter_params=[1e6, 8e6, 0.5e6],
        envelope_detector="I_Q",
        picmus_dataset=True,
        channel_reduction=args.channel_reduction,
    )

    das = BFCartesianReference(**common)
    mvbf = BFMVBFspatial(
        **common,
        window_width=args.window_width,
        apply_apodization=args.mvbf_apply_apodization,
    )
    return das, mvbf


def time_call(label, func):
    print("\nRunning {}...".format(label))
    start = time.time()
    result = func()
    elapsed = time.time() - start
    print("{} wall time: {:.3f} s".format(label, elapsed))
    return result, elapsed


def to_db(image, db_range):
    magnitude = np.abs(image).astype(np.float64)
    magnitude /= np.max(magnitude) + 1e-12
    image_db = 20 * np.log10(magnitude + 1e-12)
    return np.clip(image_db, -db_range, 0)


def image_axes(args):
    nx, nz = args.image_res
    x0, x1 = args.x_range
    z0, z1 = args.z_range
    return np.linspace(x0, x1, nx), np.linspace(z0, z1, nz)


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
        return np.nan, inner.size, outer.size, np.nan, np.nan
    inner_mean = np.mean(inner)
    outer_mean = np.mean(outer)
    denominator = 0.5 * np.sqrt(np.var(inner) + np.var(outer))
    cnr = 20 * np.log10((abs(outer_mean - inner_mean) + 1e-12) / (denominator + 1e-12))
    return float(cnr), int(inner.size), int(outer.size), float(inner_mean), float(outer_mean)


def evaluate_cnr(args, das_image, mvbf_image):
    axis_x, axis_z = image_axes(args)
    rows = []
    for circle_index, circle in enumerate(DEFAULT_CNR_CIRCLES, start=1):
        for method, image in [("DAS", das_image), ("MVBFss", mvbf_image)]:
            cnr, inner_px, outer_px, inner_mean, outer_mean = cnr_for_circle(image, axis_x, axis_z, circle)
            rows.append({
                "circle": circle_index,
                "method": method,
                "cnr_db": cnr,
                "inner_pixels": inner_px,
                "outer_pixels": outer_px,
                "inner_mean": inner_mean,
                "outer_mean": outer_mean,
                "x_m": circle[0],
                "z_m": circle[1],
                "outer_radius_m": circle[2],
                "inner_radius_m": circle[3],
            })
    return rows


def save_outputs(args, das_image, mvbf_image, das_time, mvbf_time, cnr_rows):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "aperture{}_L{}_{}x{}".format(
        args.channel_reduction,
        args.window_width,
        args.image_res[0],
        args.image_res[1],
    )
    figure_path = args.output_dir / ("picmus_das_vs_mvbfss_{}.png".format(suffix))
    csv_path = args.output_dir / ("picmus_das_vs_mvbfss_{}.csv".format(suffix))

    with csv_path.open("w", newline="") as file_obj:
        fieldnames = [
            "circle", "method", "cnr_db", "inner_pixels", "outer_pixels",
            "inner_mean", "outer_mean", "x_m", "z_m", "outer_radius_m", "inner_radius_m",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in cnr_rows:
            writer.writerow(row)

    das_db = to_db(das_image, args.db_range)
    mvbf_db = to_db(mvbf_image, args.db_range)
    diff_db = mvbf_db - das_db
    x0, x1 = args.x_range
    z0, z1 = args.z_range
    extent = [x0, x1, z1, z0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    panels = [
        (axes[0], das_db, "DAS", "gray", -args.db_range, 0),
        (axes[1], mvbf_db, "MVBFss", "gray", -args.db_range, 0),
        (axes[2], diff_db, "MVBFss - DAS, dB", "seismic", -12, 12),
    ]
    for ax, image, title, cmap, vmin, vmax in panels:
        im = ax.imshow(image, cmap=cmap, extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        if title != "MVBFss - DAS, dB":
            for circle in DEFAULT_CNR_CIRCLES:
                center_x, center_z, outer_radius, inner_radius = circle
                ax.add_patch(Circle((center_x, center_z), outer_radius, edgecolor="red", fill=False, lw=1.0))
                ax.add_patch(Circle((center_x, center_z), inner_radius, edgecolor="lime", fill=False, lw=1.0))
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "PICMUS contrast: DAS vs MVBFss, aperture={}, L={}, apod={}\nDAS {:.2f}s, MVBFss {:.2f}s".format(
            args.channel_reduction,
            args.window_width,
            args.mvbf_apply_apodization,
            das_time,
            mvbf_time,
        )
    )
    fig.savefig(str(figure_path), dpi=160)
    plt.close(fig)
    return figure_path, csv_path


def main():
    args = parse_args()
    print("PICMUS DAS vs FBSS-MVBF benchmark")
    print("Script: {}".format(Path(__file__).resolve()))
    print("Dataset: {}".format(args.dataset))
    print("Image resolution: {}".format(args.image_res))
    print("Channel reduction: {}".format(args.channel_reduction))
    print("MVBF window width L: {}".format(args.window_width))
    print("MVBF apply apodization: {}".format(args.mvbf_apply_apodization))
    print("Plane-wave indices: {}".format(args.plane_wave_indices))

    data_loader = DataLoader(str(args.dataset))
    try:
        rf_data, tx_strategy = load_rf_stack(data_loader, args.plane_wave_indices)
        print("RF data shape: {}".format(rf_data.shape))
        print("Dataset tx_strategy: {} {}".format(data_loader.tx_strategy[0], data_loader.tx_strategy[1].shape))
        print("Benchmark tx_strategy: {}".format(tx_strategy[0]))
        print("Transducer elements: {}".format(data_loader.transducer.num_of_elements))
        if args.channel_reduction > int(data_loader.transducer.num_of_elements):
            raise ValueError("channel_reduction cannot exceed transducer elements")

        das, mvbf = build_beamformers(args, data_loader, tx_strategy)
        das_image, das_time = time_call("DAS", lambda: das.beamform(rf_data, numba_active=True))
        mvbf_image, mvbf_time = time_call("MVBFss", lambda: mvbf.beamform(rf_data, numba_active=False))
    finally:
        data_loader.close_file()

    cnr_rows = evaluate_cnr(args, das_image, mvbf_image)
    print("\nCNR summary:")
    for circle_index in sorted(set(row["circle"] for row in cnr_rows)):
        das_cnr = [row["cnr_db"] for row in cnr_rows if row["circle"] == circle_index and row["method"] == "DAS"][0]
        mv_cnr = [row["cnr_db"] for row in cnr_rows if row["circle"] == circle_index and row["method"] == "MVBFss"][0]
        print("  circle {}: DAS={:.3f} dB, MVBFss={:.3f} dB, delta={:+.3f} dB".format(
            circle_index, das_cnr, mv_cnr, mv_cnr - das_cnr
        ))

    figure_path, csv_path = save_outputs(args, das_image, mvbf_image, das_time, mvbf_time, cnr_rows)
    print("\nSaved figure:")
    print("  {}".format(figure_path))
    print("Saved CSV:")
    print("  {}".format(csv_path))


if __name__ == "__main__":
    main()
