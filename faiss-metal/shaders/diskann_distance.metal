#include <metal_stdlib>
#include "DiskannDistanceParams.h"
using namespace metal;

/// Batch L2 squared distance: one query vs N candidates.
/// Each threadgroup (32 threads = 1 simdgroup) computes distance for one candidate.
/// All 32 lanes cooperatively split the dimension sum for coalesced reads.
///
/// Dispatch: grid = (N, 1, 1), threadgroup = (32, 1, 1)

kernel void diskann_batch_l2(
    device const float* query [[buffer(0)]],       // (dim,)
    device const float* candidates [[buffer(1)]],  // (n * dim,) contiguous
    device float* out_distances [[buffer(2)]],     // (n,)
    constant DiskannDistParams& params [[buffer(3)]],
    uint candidate_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    if (candidate_idx >= params.n) return;

    const uint dim = params.dim;
    device const float* cand = candidates + candidate_idx * dim;

    // All 32 lanes split dimensions: consecutive lanes read consecutive addresses
    float partial = 0.0f;
    for (uint j = lane; j < dim; j += 32) {
        float diff = query[j] - cand[j];
        partial += diff * diff;
    }

    float dist = simd_sum(partial);

    if (lane == 0) {
        out_distances[candidate_idx] = dist;
    }
}

/// Batch inner product distance: one query vs N candidates.
/// Returns negated dot product (lower = more similar).

kernel void diskann_batch_ip(
    device const float* query [[buffer(0)]],
    device const float* candidates [[buffer(1)]],
    device float* out_distances [[buffer(2)]],
    constant DiskannDistParams& params [[buffer(3)]],
    uint candidate_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    if (candidate_idx >= params.n) return;

    const uint dim = params.dim;
    device const float* cand = candidates + candidate_idx * dim;

    float partial = 0.0f;
    for (uint j = lane; j < dim; j += 32) {
        partial += query[j] * cand[j];
    }

    float dot = simd_sum(partial);

    if (lane == 0) {
        out_distances[candidate_idx] = -dot;
    }
}
