"""
Small numerical tutorial for the FBSS-MVBF calculation in PyBF.

This script mirrors the central pixel-wise operations in:
    scripts/beamformer_mvbf_spatial_smooth.py

It intentionally uses a tiny synthetic aperture so the intermediate arrays can
be printed and understood:
    delayed aperture vector
        -> overlapping subarray snapshots
        -> forward/backward covariance
        -> diagonal loading
        -> adaptive MV weights
        -> DAS versus MV pixel outputs

Run from the repository root:
    python tests/code/tutorial_mvbf_weights/main.py
    python tests/code/tutorial_mvbf_weights/main.py --loading-scale 0.1
    python tests/code/tutorial_mvbf_weights/main.py --clutter-phase-step-deg 110
"""

import argparse
import numpy as np


np.set_printoptions(precision=3, suppress=True)


def spatial_snapshots(channel_vector, subarray_length):
    """Build overlapping spatial-smoothing snapshots."""
    n_snapshots = channel_vector.shape[0] - subarray_length + 1
    snapshots = np.zeros((n_snapshots, subarray_length), dtype=np.complex128)
    for index in range(n_snapshots):
        snapshots[index, :] = channel_vector[index:index + subarray_length]
    return snapshots


def pybf_fbss_covariance(snapshots, loading_scale=1.0):
    """Mirror the forward-backward covariance construction in PyBF."""
    subarray_length = snapshots.shape[1]
    r_forward = np.zeros((subarray_length, subarray_length), dtype=np.complex128)
    r_backward = np.zeros((subarray_length, subarray_length), dtype=np.complex128)

    for snapshot in snapshots:
        r_forward += np.outer(np.conj(snapshot), snapshot)
        reversed_snapshot = np.flip(snapshot)
        r_backward += np.outer(np.conj(reversed_snapshot), reversed_snapshot)

    # loading_scale=1.0 deliberately follows the loading expression used by
    # the project. Smaller values allow stronger adaptation, but may reduce
    # numerical robustness.
    loading = loading_scale * np.identity(subarray_length) * np.trace(r_forward) / subarray_length
    r_loaded = 0.5 * r_forward + 0.5 * r_backward + loading
    return r_forward, r_backward, loading, r_loaded


def pybf_mv_weights(r_loaded):
    """Mirror PyBF's Capon/MV weight convention after delay alignment."""
    steering_vector = np.ones(r_loaded.shape[0], dtype=np.complex128)
    r_inv = np.linalg.inv(r_loaded)
    numerator = np.matmul(r_inv, steering_vector)
    denominator = np.sum(numerator)
    weights = numerator / denominator
    return steering_vector, r_inv, weights


def subarray_beamform(snapshots, weights):
    """Average the weighted outputs of all overlapping subarrays."""
    return np.mean(np.sum(snapshots * weights.reshape(1, -1), axis=1))


def print_matrix(name, array):
    print("\n{} shape={}".format(name, array.shape))
    print(array)


def evaluate_loading(snapshots, clutter_snapshots, loading_scale):
    """Compute MV clutter leakage and stability indicators for one loading."""
    _, _, _, r_loaded = pybf_fbss_covariance(snapshots, loading_scale)
    _, _, weights = pybf_mv_weights(r_loaded)
    clutter_leakage = abs(subarray_beamform(clutter_snapshots, weights))
    return clutter_leakage, np.linalg.cond(r_loaded), np.max(np.abs(weights))


def main():
    parser = argparse.ArgumentParser(description="Tiny FBSS-MVBF covariance and weights tutorial.")
    parser.add_argument(
        "--loading-scale",
        type=float,
        default=1.0,
        help="Diagonal loading factor. 1.0 matches beamformer_mvbf_spatial_smooth.py.",
    )
    parser.add_argument(
        "--clutter-phase-step-deg",
        type=float,
        default=70.0,
        help="Spatial phase increment of the synthetic clutter component.",
    )
    parser.add_argument(
        "--sweep-loading",
        nargs="+",
        type=float,
        default=[0.0, 0.001, 0.01, 0.1, 1.0, 10.0],
        help="Loading factors summarized after the detailed single-case output.",
    )
    args = parser.parse_args()

    # After correct delay alignment, a target at the current pixel is modeled
    # as equal phase across receiving channels.
    n_channels = 8
    subarray_length = 4
    target = np.ones(n_channels, dtype=np.complex128)

    # A residual off-axis/clutter component has a spatial phase slope across
    # the aperture. It is not consistent with the all-ones target steering
    # response after delay alignment.
    clutter_phase_step = np.radians(args.clutter_phase_step_deg)
    clutter = 1.3 * np.exp(1.j * clutter_phase_step * np.arange(n_channels))

    delayed_pixel_data = target + clutter

    print("PyBF FBSS-MVBF tutorial: one synthetic pixel")
    print("=" * 68)
    print("Number of selected channels (M):", n_channels)
    print("Subarray length (L):", subarray_length)
    print("Number of overlapping snapshots (K):", n_channels - subarray_length + 1)
    print("Clutter spatial phase step: {:.1f} deg".format(args.clutter_phase_step_deg))
    print("Diagonal loading scale: {} (1.0 matches PyBF code)".format(args.loading_scale))
    print("\nInterpretation:")
    print("  target  = desired echo after delay alignment (all channels coherent)")
    print("  clutter = residual spatially varying component to suppress")

    print_matrix("target channel vector", target)
    print_matrix("clutter channel vector", clutter)
    print_matrix("delayed target + clutter vector", delayed_pixel_data)

    snapshots = spatial_snapshots(delayed_pixel_data, subarray_length)
    target_snapshots = spatial_snapshots(target, subarray_length)
    clutter_snapshots = spatial_snapshots(clutter, subarray_length)
    print_matrix("overlapping snapshots S", snapshots)

    r_forward, r_backward, loading, r_loaded = pybf_fbss_covariance(snapshots, args.loading_scale)
    print_matrix("forward covariance R_f", r_forward)
    print_matrix("backward covariance R_b", r_backward)
    print_matrix("diagonal loading", loading)
    print_matrix("loaded FBSS covariance R_loaded", r_loaded)

    steering_vector, r_inv, mv_weights = pybf_mv_weights(r_loaded)
    das_weights = np.ones(subarray_length, dtype=np.complex128) / subarray_length

    print_matrix("steering vector a", steering_vector)
    print_matrix("inverse covariance R_loaded^-1", r_inv)
    print_matrix("fixed DAS weights", das_weights)
    print_matrix("adaptive MV weights", mv_weights)
    print("\nWeight constraint check: sum(MV weights) = {}".format(np.sum(mv_weights)))

    das_target = subarray_beamform(target_snapshots, das_weights)
    das_clutter = subarray_beamform(clutter_snapshots, das_weights)
    das_combined = subarray_beamform(snapshots, das_weights)

    mv_target = subarray_beamform(target_snapshots, mv_weights)
    mv_clutter = subarray_beamform(clutter_snapshots, mv_weights)
    mv_combined = subarray_beamform(snapshots, mv_weights)

    print("\n" + "=" * 68)
    print("Output decomposition for this pixel")
    print("                         DAS                     MVBF")
    print("Target response      {:>20} {:>20}".format(das_target, mv_target))
    print("Clutter leakage      {:>20} {:>20}".format(das_clutter, mv_clutter))
    print("Combined output      {:>20} {:>20}".format(das_combined, mv_combined))
    print("|clutter leakage|    {:>20.6f} {:>20.6f}".format(abs(das_clutter), abs(mv_clutter)))

    print("\n" + "=" * 68)
    print("Diagonal loading sweep")
    print("DAS reference |clutter leakage| = {:.6f}".format(abs(das_clutter)))
    print("scale       |MV clutter|   DAS/MV reduction   cond(R_loaded)   max|w|")
    for loading_scale in args.sweep_loading:
        clutter_leakage, condition_number, max_weight = evaluate_loading(
            snapshots,
            clutter_snapshots,
            loading_scale,
        )
        reduction = abs(das_clutter) / max(clutter_leakage, 1e-12)
        print("{:>10g}   {:>12.6f}   {:>16.3f}   {:>14.3f}   {:>7.3f}".format(
            loading_scale,
            clutter_leakage,
            reduction,
            condition_number,
            max_weight,
        ))

    print("\nTakeaway:")
    print("  Both DAS and MVBF preserve the aligned target approximately.")
    print("  DAS uses fixed uniform weights in this small demonstration.")
    print("  MVBF changes complex channel weights using covariance evidence.")
    print("  Less loading typically increases adaptation, but can make weights")
    print("  more sensitive when the covariance matrix is poorly conditioned.")
    print("  If delay/aberration correction is wrong, the real target may no")
    print("  longer match the all-ones steering vector and MVBF may suppress it.")


if __name__ == "__main__":
    main()
