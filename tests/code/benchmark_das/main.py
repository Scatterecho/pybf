"""
Benchmark DAS kernel implementations used by PyBF.

This script compares:
    1. delay_and_sum_numpy
    2. delay_and_sum_numba

It intentionally benchmarks the low-level DAS kernels instead of the full
BFCartesianRealTime.beamform() pipeline, so that RF preprocessing and plotting
do not dominate the result.

The benchmark can simulate multiple transmit modes, such as multi-angle
plane-wave compounding, by building a delays_idx tensor with shape:

    (n_modes, n_elements, n_points)

Note:
    The NumPy implementation uses fancy indexing and can allocate very large
    temporary arrays. Keep image_res modest when testing many modes.
"""

from pathlib import Path
import argparse
import gc
import statistics
import sys
import time

import numpy as np


# Make the local pybf package importable when this script is launched directly.
# File location:
#   <repo>/tests/code/benchmark_das/main.py
# Import root required by this project layout:
#   <repo_parent>
PATH_TO_LIB = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PATH_TO_LIB))

from pybf.pybf.image_settings import ImageSettings
from pybf.pybf.transducer import Transducer
from pybf.pybf.bf_cores import delay_and_sum_numpy, delay_and_sum_numba
from pybf.scripts.beamformer_cartesian_realtime import BFCartesianRealTime


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


def estimate_numpy_temp_mb(n_modes, n_points, n_elements):
    """Rough peak-memory estimate for delay_and_sum_numpy temporary arrays.

    The NumPy kernel materializes channel/sample fancy-index arrays and a
    delayed RF tensor. During apodization it may also allocate another
    multiplied tensor. This is intentionally conservative.
    """
    bytes_per_entry = 4 + 8 + 8 + 8  # sample idx + channel idx + delayed + multiply temp
    return n_modes * n_points * n_elements * bytes_per_entry / (1024 ** 2)


def make_kernel_inputs(n_modes, image_res, max_angle):
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

    tx_strategy = [f"PW_{n_modes}_{max_angle}", [0]]

    bf = BFCartesianRealTime(
        20e6,
        tx_strategy,
        trans,
        decimation_factor=1,
        interpolation_factor=10,
        image_res=image_res,
        img_config_obj=img_config,
        start_time=-1.0666666666666665e-06,
        correction_time_shift=2.6e-06,
        alpha_fov_apod=40,
        bp_filter_params=None,
    )

    rf_path = PATH_TO_LIB / "pybf" / "tests" / "data" / "sample_rf_data.csv"
    rf_data = np.genfromtxt(str(rf_path), delimiter=",")

    # Preprocess once. This benchmark focuses on DAS kernels, not preprocessing.
    rf_data_proc = bf._preprocess_data(rf_data)
    rf_data_proc_trans = np.ascontiguousarray(np.transpose(rf_data_proc))

    # Build delays_idx with shape (n_modes, n_elements, n_points).
    rx = bf._rx_delays_samples.reshape(1, bf._rx_delays_samples.shape[0], -1)
    tx = bf._tx_delays_samples.reshape(n_modes, 1, -1)
    delays_idx = np.ascontiguousarray(rx + tx, dtype=np.int32)

    apod = np.ascontiguousarray(bf._apod, dtype=np.float32)

    return rf_data_proc_trans, delays_idx, apod


def time_call(fn, rf_data, delays_idx_base, apod):
    # delay_and_sum_numpy mutates delays_idx in-place when clipping out-of-range
    # indices, so each timing run receives a fresh copy. The copy itself is not
    # included in the measured kernel time.
    delays_idx = delays_idx_base.copy()
    gc.collect()
    start = time.perf_counter()
    out = fn(rf_data, delays_idx, apod_weights=apod)
    elapsed = time.perf_counter() - start
    return elapsed, out


def summarize(times):
    return {
        "min": min(times),
        "median": statistics.median(times),
        "mean": statistics.mean(times),
    }


def run_one_case(n_modes, image_res, repeats, max_angle, max_temp_mb):
    rf_data, delays_idx, apod = make_kernel_inputs(n_modes, image_res, max_angle)

    n_samples, n_elements = rf_data.shape
    _, _, n_points = delays_idx.shape
    numpy_temp_mb = estimate_numpy_temp_mb(n_modes, n_points, n_elements)

    print("\n" + "=" * 78)
    print(f"Case: n_modes={n_modes}, image_res={image_res}, n_points={n_points}")
    print(f"rf_data: {rf_data.shape}, delays_idx: {delays_idx.shape}, apod: {apod.shape}")
    print(f"Estimated NumPy temporary memory: {numpy_temp_mb:,.1f} MB")

    if numpy_temp_mb > max_temp_mb:
        print(
            f"SKIP: estimated NumPy temporary memory exceeds limit "
            f"({numpy_temp_mb:,.1f} MB > {max_temp_mb:,.1f} MB)."
        )
        return None

    print("Warming up Numba JIT...")
    _, numba_warm = time_call(delay_and_sum_numba, rf_data, delays_idx, apod)

    print("Running one NumPy call for correctness reference...")
    _, numpy_ref = time_call(delay_and_sum_numpy, rf_data, delays_idx, apod)

    max_abs_err = float(np.max(np.abs(numpy_ref - numba_warm)))
    rel_err = max_abs_err / max(float(np.max(np.abs(numpy_ref))), 1e-12)
    print(f"Correctness check: max_abs_err={max_abs_err:.6g}, rel_err={rel_err:.6g}")

    numpy_times = []
    numba_times = []

    for _ in range(repeats):
        elapsed, _ = time_call(delay_and_sum_numpy, rf_data, delays_idx, apod)
        numpy_times.append(elapsed)

    for _ in range(repeats):
        elapsed, _ = time_call(delay_and_sum_numba, rf_data, delays_idx, apod)
        numba_times.append(elapsed)

    numpy_stats = summarize(numpy_times)
    numba_stats = summarize(numba_times)
    speedup = numpy_stats["median"] / numba_stats["median"]

    print("Timing summary, seconds:")
    print(f"  NumPy: min={numpy_stats['min']:.4f}, median={numpy_stats['median']:.4f}, mean={numpy_stats['mean']:.4f}")
    print(f"  Numba: min={numba_stats['min']:.4f}, median={numba_stats['median']:.4f}, mean={numba_stats['mean']:.4f}")
    print(f"  Median speedup NumPy/Numba: {speedup:.2f}x")

    return {
        "n_modes": n_modes,
        "n_points": n_points,
        "n_elements": n_elements,
        "numpy_temp_mb": numpy_temp_mb,
        "numpy_median_s": numpy_stats["median"],
        "numba_median_s": numba_stats["median"],
        "speedup": speedup,
        "max_abs_err": max_abs_err,
        "rel_err": rel_err,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyBF DAS kernels.")
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 3, 5, 9])
    parser.add_argument("--image-res", nargs=2, type=int, default=[120, 180], metavar=("X", "Z"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-angle", type=float, default=12.0)
    parser.add_argument(
        "--max-temp-mb",
        type=float,
        default=1500.0,
        help="Skip NumPy benchmark cases whose estimated temporary memory exceeds this limit.",
    )
    args = parser.parse_args()

    print("PyBF DAS kernel benchmark")
    print(f"PATH_TO_LIB: {PATH_TO_LIB}")
    print(f"Modes: {args.modes}")
    print(f"Image resolution: {args.image_res}")
    print(f"Repeats: {args.repeats}")

    results = []
    for n_modes in args.modes:
        result = run_one_case(
            n_modes=n_modes,
            image_res=args.image_res,
            repeats=args.repeats,
            max_angle=args.max_angle,
            max_temp_mb=args.max_temp_mb,
        )
        if result is not None:
            results.append(result)

    if results:
        print("\n" + "=" * 78)
        print("Compact summary")
        print("modes | points | NumPy med(s) | Numba med(s) | speedup | NumPy temp MB | rel err")
        for r in results:
            print(
                f"{r['n_modes']:>5} | "
                f"{r['n_points']:>6} | "
                f"{r['numpy_median_s']:>12.4f} | "
                f"{r['numba_median_s']:>12.4f} | "
                f"{r['speedup']:>7.2f} | "
                f"{r['numpy_temp_mb']:>13.1f} | "
                f"{r['rel_err']:.3g}"
            )


if __name__ == "__main__":
    main()
