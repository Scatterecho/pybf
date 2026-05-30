"""
Adaptive full-image MVBF experiment for PICMUS contrast/speckle data.

This script is deliberately self-contained so it can be studied without
modifying the original PyBF beamformer classes.  It implements a basic
literature-style MVBF pipeline:

    1. Run PyBF's reference DAS once to obtain delayed channel data.
    2. For each image pixel, select a pixel-dependent receive aperture from
       the FOV/F-number apodization mask.
    3. Optionally cap the local aperture by keeping the channels closest to
       the pixel lateral position.
    4. Estimate a local covariance matrix using:
          - overlapping spatial subarrays of length L
          - all selected plane waves / emissions
          - axial neighboring pixels at the same lateral coordinate
          - optional forward-backward averaging
          - diagonal loading
    5. Compute Capon/MV weights and form the pixel value.

Run from repository root, for a quick smoke test:
    python tests/code/adaptive_mvbf_full_image/main.py --image-res 40 60

A more useful low-resolution experiment:
    python tests/code/adaptive_mvbf_full_image/main.py --image-res 80 120 --max-channels 64 --subarray-length 8 --axial-half-window 1
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


DEFAULT_DATASET = PATH_TO_LIB / "pybf" / "tests" / "data" / "Picmus" / "contrast_speckle" / "rf_dataset.hdf5"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

DEFAULT_CNR_CIRCLES = np.asarray([
    [-0.00043, 0.01492, 0.0035, 0.00172],
    [-0.00043, 0.04279, 0.0035, 0.00172],
    [-0.00720, 0.02829, 0.0063, 0.00315],
], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive full-image MVBF on PICMUS data")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-res", type=int, nargs=2, default=[40, 60], metavar=("NX", "NZ"))
    parser.add_argument("--x-range", type=float, nargs=2, default=[-0.019, 0.019], metavar=("X0", "X1"))
    parser.add_argument("--z-range", type=float, nargs=2, default=[0.005, 0.050], metavar=("Z0", "Z1"))
    parser.add_argument("--plane-wave-indices", type=int, nargs="+", default=[33, 37, 38, 42])
    parser.add_argument("--max-channels", type=int, default=64, help="Maximum local receive aperture size per pixel")
    parser.add_argument("--subarray-length", type=int, default=8, help="Spatial smoothing subarray length L")
    parser.add_argument("--axial-half-window", type=int, default=1, help="Use +/- this many axial pixels for covariance averaging")
    parser.add_argument("--diagonal-loading", type=float, default=0.03, help="Diagonal loading as fraction of trace(R)/L")
    parser.add_argument("--alpha-fov-apod", type=float, default=40.0)
    parser.add_argument("--interpolation-factor", type=int, default=10)
    parser.add_argument("--decimation-factor", type=int, default=1)
    parser.add_argument("--db-range", type=float, default=50.0)
    parser.add_argument("--no-forward-backward", dest="forward_backward", action="store_false", default=True)
    parser.add_argument("--no-local-hanning", dest="local_hanning", action="store_false", default=True)
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


def build_das_reference(args, data_loader, tx_strategy):
    x0, x1 = args.x_range
    z0, z1 = args.z_range
    img_config = ImageSettings(x0, x1, z0, z1, 5, data_loader.transducer)
    return BFCartesianReference(
        data_loader.f_sampling,
        tx_strategy,
        data_loader.transducer,
        args.decimation_factor,
        args.interpolation_factor,
        args.image_res,
        img_config,
        start_time=0,
        correction_time_shift=0,
        alpha_fov_apod=args.alpha_fov_apod,
        bp_filter_params=[1e6, 8e6, 0.5e6],
        envelope_detector="I_Q",
        picmus_dataset=True,
        channel_reduction=int(data_loader.transducer.num_of_elements),
    )


def select_local_channels(apod_row, elements_x, pixel_x, max_channels):
    active = np.where(apod_row > 0)[0]
    if active.size == 0:
        return active
    if max_channels is not None and active.size > max_channels:
        order = np.argsort(np.abs(elements_x[active] - pixel_x))
        active = np.sort(active[order[:max_channels]])
    return active


def accumulate_covariance(delayed_stack, apod, pixel_index, nx, nz, active_channels, subarray_length,
                          axial_half_window, local_hanning, diagonal_loading, forward_backward):
    L = subarray_length
    n_modes = delayed_stack.shape[0]
    z_index = pixel_index // nx
    x_index = pixel_index % nx

    R = np.zeros((L, L), dtype=np.complex128)
    snapshot_count = 0

    for dz in range(-axial_half_window, axial_half_window + 1):
        z_neighbor = z_index + dz
        if z_neighbor < 0 or z_neighbor >= nz:
            continue
        neighbor_index = z_neighbor * nx + x_index
        neighbor_apod = apod[neighbor_index, active_channels]

        for mode_index in range(n_modes):
            vector = delayed_stack[mode_index, neighbor_index, active_channels].astype(np.complex128)
            if local_hanning:
                # Re-window the local aperture smoothly.  This avoids the hard
                # channel-set discontinuity being the only aperture weighting.
                vector = vector * np.hanning(vector.size)
            else:
                vector = vector * neighbor_apod

            if vector.size < L:
                continue

            for start in range(0, vector.size - L + 1):
                sub = vector[start:start + L]
                R += np.outer(np.conj(sub), sub)
                snapshot_count += 1
                if forward_backward:
                    sub_b = np.flip(sub)
                    R += np.outer(np.conj(sub_b), sub_b)
                    snapshot_count += 1

    if snapshot_count == 0:
        return None, 0

    R /= float(snapshot_count)
    trace_R = np.real(np.trace(R))
    if trace_R <= 0:
        return None, snapshot_count
    R += diagonal_loading * trace_R / L * np.eye(L, dtype=np.complex128)
    return R, snapshot_count


def mvbf_pixel_output(delayed_stack, pixel_index, active_channels, weights, subarray_length, local_hanning):
    L = subarray_length
    n_modes = delayed_stack.shape[0]
    pixel_value = 0.0 + 0.0j

    for mode_index in range(n_modes):
        vector = delayed_stack[mode_index, pixel_index, active_channels].astype(np.complex128)
        if local_hanning:
            vector = vector * np.hanning(vector.size)
        if vector.size < L:
            continue
        x_sum = np.zeros(L, dtype=np.complex128)
        n_snapshots = vector.size - L + 1
        for start in range(0, n_snapshots):
            x_sum += vector[start:start + L]
        pixel_value += np.sum(weights.T * x_sum) / float(n_snapshots)

    return pixel_value


def adaptive_mvbf(delayed_stack, apod, pixels_coords, elements_x, image_res, args):
    nx, nz = image_res
    n_points = nx * nz
    output = np.zeros(n_points, dtype=np.complex128)
    snapshot_counts = np.zeros(n_points, dtype=np.int32)
    active_counts = np.zeros(n_points, dtype=np.int32)

    start_time = time.time()
    for pixel_index in range(n_points):
        active = select_local_channels(
            apod[pixel_index, :],
            elements_x,
            pixels_coords[0, pixel_index],
            args.max_channels,
        )
        active_counts[pixel_index] = active.size
        if active.size < args.subarray_length:
            continue

        R, snapshot_count = accumulate_covariance(
            delayed_stack,
            apod,
            pixel_index,
            nx,
            nz,
            active,
            args.subarray_length,
            args.axial_half_window,
            args.local_hanning,
            args.diagonal_loading,
            args.forward_backward,
        )
        snapshot_counts[pixel_index] = snapshot_count
        if R is None:
            continue

        steering = np.ones(args.subarray_length, dtype=np.complex128)
        # solve() is numerically cleaner than explicitly inverting R.
        numerator = np.linalg.solve(R, steering)
        denominator = np.sum(numerator)
        if np.abs(denominator) < 1e-12:
            continue
        weights = numerator / denominator
        output[pixel_index] = mvbf_pixel_output(
            delayed_stack,
            pixel_index,
            active,
            weights,
            args.subarray_length,
            args.local_hanning,
        )

    elapsed = time.time() - start_time
    return output.reshape(nz, nx), elapsed, active_counts.reshape(nz, nx), snapshot_counts.reshape(nz, nx)


def to_db(image, db_range):
    mag = np.abs(image).astype(np.float64)
    mag /= np.max(mag) + 1e-12
    db = 20 * np.log10(mag + 1e-12)
    return np.clip(db, -db_range, 0)


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
        return np.nan
    denominator = 0.5 * np.sqrt(np.var(inner) + np.var(outer))
    return float(20 * np.log10((abs(np.mean(outer) - np.mean(inner)) + 1e-12) / (denominator + 1e-12)))


def evaluate_cnr(args, das_image, mvbf_image):
    axis_x, axis_z = image_axes(args)
    rows = []
    for circle_index, circle in enumerate(DEFAULT_CNR_CIRCLES, start=1):
        das_cnr = cnr_for_circle(das_image, axis_x, axis_z, circle)
        mvbf_cnr = cnr_for_circle(mvbf_image, axis_x, axis_z, circle)
        rows.append({
            "circle": circle_index,
            "das_cnr_db": das_cnr,
            "adaptive_mvbf_cnr_db": mvbf_cnr,
            "delta_db": mvbf_cnr - das_cnr,
        })
    return rows


def save_outputs(args, das_image, mvbf_image, active_counts, snapshot_counts, das_time, mvbf_time, cnr_rows):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "{}x{}_maxch{}_L{}_ax{}_dl{}".format(
        args.image_res[0],
        args.image_res[1],
        args.max_channels,
        args.subarray_length,
        args.axial_half_window,
        str(args.diagonal_loading).replace(".", "p"),
    )
    figure_path = args.output_dir / ("adaptive_mvbf_{}.png".format(suffix))
    csv_path = args.output_dir / ("adaptive_mvbf_{}.csv".format(suffix))

    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["circle", "das_cnr_db", "adaptive_mvbf_cnr_db", "delta_db"])
        writer.writeheader()
        for row in cnr_rows:
            writer.writerow(row)

    das_db = to_db(das_image, args.db_range)
    mvbf_db = to_db(mvbf_image, args.db_range)
    diff_db = np.clip(mvbf_db - das_db, -12, 12)
    x0, x1 = args.x_range
    z0, z1 = args.z_range
    extent = [x0, x1, z1, z0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    panels = [
        (axes[0], das_db, "DAS reference", "gray", -args.db_range, 0),
        (axes[1], mvbf_db, "Adaptive MVBF", "gray", -args.db_range, 0),
        (axes[2], diff_db, "MVBF - DAS, dB", "seismic", -12, 12),
        (axes[3], active_counts, "local active channels", "viridis", 0, max(args.max_channels, 1)),
    ]
    for ax, image, title, cmap, vmin, vmax in panels:
        im = ax.imshow(image, cmap=cmap, extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        if title in ["DAS reference", "Adaptive MVBF"]:
            for circle in DEFAULT_CNR_CIRCLES:
                center_x, center_z, outer_radius, inner_radius = circle
                ax.add_patch(Circle((center_x, center_z), outer_radius, edgecolor="red", fill=False, lw=1.0))
                ax.add_patch(Circle((center_x, center_z), inner_radius, edgecolor="lime", fill=False, lw=1.0))
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "Adaptive MVBF: maxch={}, L={}, axial +/-{}, DL={}, FB={}, local Hann={}\nDAS {:.2f}s, MVBF {:.2f}s".format(
            args.max_channels,
            args.subarray_length,
            args.axial_half_window,
            args.diagonal_loading,
            args.forward_backward,
            args.local_hanning,
            das_time,
            mvbf_time,
        )
    )
    fig.savefig(str(figure_path), dpi=160)
    plt.close(fig)
    return figure_path, csv_path


def main():
    args = parse_args()
    print("Adaptive full-image MVBF experiment")
    print("Script: {}".format(Path(__file__).resolve()))
    print("Dataset: {}".format(args.dataset))
    print("Image resolution: {}".format(args.image_res))
    print("max_channels={}, L={}, axial_half_window={}, diagonal_loading={}".format(
        args.max_channels,
        args.subarray_length,
        args.axial_half_window,
        args.diagonal_loading,
    ))
    print("forward_backward={}, local_hanning={}".format(args.forward_backward, args.local_hanning))

    data_loader = DataLoader(str(args.dataset))
    try:
        rf_data, tx_strategy = load_rf_stack(data_loader, args.plane_wave_indices)
        print("RF data shape: {}".format(rf_data.shape))
        print("tx_strategy: {}".format(tx_strategy[0]))

        das = build_das_reference(args, data_loader, tx_strategy)
        start = time.time()
        das_image = das.beamform(rf_data, numba_active=True)
        das_time = time.time() - start
        delayed_stack = das.bf_data
        print("Delayed stack shape from DAS reference: {}".format(delayed_stack.shape))

        mvbf_image, mvbf_time, active_counts, snapshot_counts = adaptive_mvbf(
            delayed_stack,
            das._apod,
            das._pixels_coords,
            data_loader.transducer.elements_coords[0, :],
            args.image_res,
            args,
        )
    finally:
        data_loader.close_file()

    print("DAS wall time: {:.3f} s".format(das_time))
    print("Adaptive MVBF wall time: {:.3f} s".format(mvbf_time))
    print("Active channel count: min={}, median={}, max={}".format(
        int(np.min(active_counts)),
        float(np.median(active_counts)),
        int(np.max(active_counts)),
    ))
    print("Snapshot count: min={}, median={}, max={}".format(
        int(np.min(snapshot_counts)),
        float(np.median(snapshot_counts)),
        int(np.max(snapshot_counts)),
    ))

    cnr_rows = evaluate_cnr(args, das_image, mvbf_image)
    print("\nCNR summary:")
    for row in cnr_rows:
        print("  circle {circle}: DAS={das_cnr_db:.3f} dB, adaptive MVBF={adaptive_mvbf_cnr_db:.3f} dB, delta={delta_db:+.3f} dB".format(**row))

    figure_path, csv_path = save_outputs(args, das_image, mvbf_image, active_counts, snapshot_counts, das_time, mvbf_time, cnr_rows)
    print("\nSaved figure:")
    print("  {}".format(figure_path))
    print("Saved CSV:")
    print("  {}".format(csv_path))


if __name__ == "__main__":
    main()
