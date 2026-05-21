"""
Benchmark full-frame PyBF beamforming paths:

1. Analytic-RF path used by the original realtime demo:
       RF -> IQ demodulate -> interpolate -> remodulate -> DAS

2. Direct-IQ path added for engineering experiments:
       RF -> IQ demodulate -> DAS on IQ with delay-dependent phase rotation

The visual comparison image is saved by default to:
    tests/code/benchmark_iq_phase_rotator/iq_phase_rotator_comparison.png
"""

from pathlib import Path
import argparse
import gc
import statistics
import sys
import time

import numpy as np
import psutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# File location:
#   <repo>/tests/code/benchmark_iq_phase_rotator/main.py
# Import root required by this project layout:
#   <repo_parent>
PATH_TO_LIB = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PATH_TO_LIB))

from pybf.pybf.image_settings import ImageSettings
from pybf.pybf.transducer import Transducer
from pybf.scripts.beamformer_cartesian_realtime import BFCartesianRealTime


RF_PATH = PATH_TO_LIB / "pybf" / "tests" / "data" / "sample_rf_data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "iq_phase_rotator_comparison.png"


def build_transducer():
    return Transducer(
        num_of_x_elements=192,
        num_of_y_elements=1,
        x_pitch=0.0003,
        y_pitch=0,
        x_width=0,
        y_width=0,
        f_central_hz=5.2083e6,
        bandwidth_hz=5.2083e6,
        active_elements=None,
    )


def build_beamformer(image_res, decimation_factor, interpolation_factor, reconstruction_sos, tx_nominal_sos):
    trans = build_transducer()
    image_x_range = [-0.022, 0.022]
    image_z_range = [0.0014784, 0.04]

    img_config = ImageSettings(
        image_x_range[0],
        image_x_range[1],
        image_z_range[0],
        image_z_range[1],
        5,
        trans,
    )

    bf = BFCartesianRealTime(
        20e6,
        ["PW_1_0", [0]],
        trans,
        decimation_factor=decimation_factor,
        interpolation_factor=interpolation_factor,
        image_res=image_res,
        img_config_obj=img_config,
        start_time=-1.0666666666666665e-06,
        correction_time_shift=2.6e-06,
        alpha_fov_apod=40,
        bp_filter_params=None,
        reconstruction_speed_of_sound=reconstruction_sos,
        tx_nominal_speed_of_sound=tx_nominal_sos,
    )
    return bf, image_x_range, image_z_range


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)


def summarize(times):
    return {
        "min": min(times),
        "median": statistics.median(times),
        "mean": statistics.mean(times),
    }


def measure_path(label, bf, rf_data, repeats, iq_phase_correction):
    print("\n" + "=" * 78)
    print("Path:", label)
    print("Warmup run, includes any JIT compilation needed by this path...")
    _ = bf.beamform(rf_data, numba_active=True, iq_phase_correction=iq_phase_correction)

    times = []
    rss_deltas = []
    last_img = None

    for r in range(repeats):
        gc.collect()
        before = rss_mb()
        start = time.perf_counter()
        last_img = bf.beamform(rf_data, numba_active=True, iq_phase_correction=iq_phase_correction)
        elapsed = time.perf_counter() - start
        after = rss_mb()
        times.append(elapsed)
        rss_deltas.append(after - before)
        print("  repeat {}: {:.4f} s, RSS delta {:+.1f} MB".format(r + 1, elapsed, after - before))

    stats = summarize(times)
    print("Summary: min={:.4f}s, median={:.4f}s, mean={:.4f}s".format(
        stats["min"], stats["median"], stats["mean"]
    ))

    return last_img, stats, rss_deltas


def normalized_db(img, db_range=50):
    mag = np.abs(img).astype(np.float64)
    mag = mag / max(float(np.max(mag)), 1e-12)
    db = 20 * np.log10(mag + 1e-12)
    return np.clip(db, -db_range, 0)


def image_quality_metrics(ref_img, test_img):
    ref_complex = ref_img.astype(np.complex128)
    test_complex = test_img.astype(np.complex128)

    ref_mag = np.abs(ref_complex)
    test_mag = np.abs(test_complex)

    rel_l2_complex = np.linalg.norm(test_complex - ref_complex) / max(np.linalg.norm(ref_complex), 1e-12)
    rel_l2_mag = np.linalg.norm(test_mag - ref_mag) / max(np.linalg.norm(ref_mag), 1e-12)

    ref_db = normalized_db(ref_img)
    test_db = normalized_db(test_img)
    diff_db = test_db - ref_db

    corr_db = np.corrcoef(ref_db.ravel(), test_db.ravel())[0, 1]
    mae_db = np.mean(np.abs(diff_db))
    p95_abs_db = np.percentile(np.abs(diff_db), 95)

    return {
        "rel_l2_complex": float(rel_l2_complex),
        "rel_l2_mag": float(rel_l2_mag),
        "corr_db": float(corr_db),
        "mae_db": float(mae_db),
        "p95_abs_db": float(p95_abs_db),
        "ref_db": ref_db,
        "test_db": test_db,
        "diff_db": diff_db,
    }


def save_comparison_figure(ref_db, iq_db, diff_db, image_x_range, image_z_range, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extent = [image_x_range[0], image_x_range[1], image_z_range[1], image_z_range[0]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    im0 = axes[0].imshow(ref_db, cmap="gray", vmin=-50, vmax=0, extent=extent, aspect="auto")
    axes[0].set_title("Analytic RF DAS reference")
    axes[0].set_xlabel("x, m")
    axes[0].set_ylabel("z, m")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(iq_db, cmap="gray", vmin=-50, vmax=0, extent=extent, aspect="auto")
    axes[1].set_title("IQ + phase rotator DAS")
    axes[1].set_xlabel("x, m")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff_db, cmap="bwr", vmin=-12, vmax=12, extent=extent, aspect="auto")
    axes[2].set_title("Difference, IQ - reference, dB")
    axes[2].set_xlabel("x, m")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)
    return output_path


def estimate_preprocessed_memory_mb(rf_data, decimation_factor, interpolation_factor):
    n_channels, n_samples = rf_data.shape
    iq_samples = int(np.ceil(float(n_samples) / decimation_factor))
    analytic_rf_samples = iq_samples * interpolation_factor

    # Observed dtypes in this project:
    #   demodulate_decimate -> complex64
    #   interpolate_modulate -> complex128
    iq_mb = n_channels * iq_samples * np.dtype(np.complex64).itemsize / (1024 ** 2)
    analytic_rf_mb = n_channels * analytic_rf_samples * np.dtype(np.complex128).itemsize / (1024 ** 2)
    return analytic_rf_mb, iq_mb


def main():
    parser = argparse.ArgumentParser(description="Benchmark analytic-RF DAS vs direct IQ + phase-rotator DAS.")
    parser.add_argument("--image-res", nargs=2, type=int, default=[120, 180], metavar=("X", "Z"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decimation-factor", type=int, default=1)
    parser.add_argument("--interpolation-factor", type=int, default=10)
    parser.add_argument("--reconstruction-sos", type=float, default=1540.0)
    parser.add_argument("--tx-nominal-sos", type=float, default=1540.0)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    print("PyBF full-frame benchmark: analytic RF vs IQ + phase rotator")
    print("Script:", Path(__file__).resolve())
    print("RF data:", RF_PATH)
    print("Image resolution:", args.image_res)
    print("Repeats:", args.repeats)

    rf_data = np.genfromtxt(str(RF_PATH), delimiter=",")
    analytic_rf_mb, iq_mb = estimate_preprocessed_memory_mb(
        rf_data,
        args.decimation_factor,
        args.interpolation_factor,
    )

    print("\nEstimated per-frame preprocessed signal array memory:")
    print("  Analytic RF path: {:.1f} MB".format(analytic_rf_mb))
    print("  Direct IQ path:   {:.1f} MB".format(iq_mb))
    print("  Ratio:            {:.1f}x".format(analytic_rf_mb / max(iq_mb, 1e-12)))

    bf, image_x_range, image_z_range = build_beamformer(
        image_res=args.image_res,
        decimation_factor=args.decimation_factor,
        interpolation_factor=args.interpolation_factor,
        reconstruction_sos=args.reconstruction_sos,
        tx_nominal_sos=args.tx_nominal_sos,
    )

    ref_img, ref_stats, _ = measure_path(
        "Analytic RF path, RF -> IQ -> interpolate -> remodulate -> Numba DAS",
        bf,
        rf_data,
        args.repeats,
        iq_phase_correction=False,
    )

    iq_img, iq_stats, _ = measure_path(
        "Direct IQ path, RF -> IQ -> phase-rotator Numba DAS",
        bf,
        rf_data,
        args.repeats,
        iq_phase_correction=True,
    )

    metrics = image_quality_metrics(ref_img, iq_img)
    output_path = save_comparison_figure(
        metrics["ref_db"],
        metrics["test_db"],
        metrics["diff_db"],
        image_x_range,
        image_z_range,
        args.output,
    )

    print("\n" + "=" * 78)
    print("Compact summary")
    print("  Analytic RF median time: {:.4f} s".format(ref_stats["median"]))
    print("  Direct IQ median time:   {:.4f} s".format(iq_stats["median"]))
    print("  Speedup RF/IQ:           {:.2f}x".format(ref_stats["median"] / max(iq_stats["median"], 1e-12)))
    print("\nImage similarity, analytic RF path used as reference:")
    print("  Complex relative L2 error: {:.4g}".format(metrics["rel_l2_complex"]))
    print("  Magnitude relative L2 error: {:.4g}".format(metrics["rel_l2_mag"]))
    print("  dB image correlation:        {:.4f}".format(metrics["corr_db"]))
    print("  Mean absolute dB diff:       {:.4f} dB".format(metrics["mae_db"]))
    print("  95th percentile |dB diff|:   {:.4f} dB".format(metrics["p95_abs_db"]))
    print("\nSaved comparison figure:")
    print("  {}".format(output_path))


if __name__ == "__main__":
    main()
