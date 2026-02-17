//! FFI bridge to Metal GPU batch distance computation.
//!
//! The actual Metal implementation lives in C++/ObjC++ (metal_diskann_bridge.mm).
//! On non-macOS platforms, stub functions return "not available".
//! Symbols are resolved at link time when the Rust static lib is linked
//! with the C++ extension.

use std::sync::atomic::{AtomicI32, Ordering};

extern "C" {
    fn diskann_metal_available() -> i32;
    fn diskann_metal_batch_distances(
        query: *const f32,
        candidates: *const f32,
        n: i32,
        dim: i32,
        metric: i32,
        out_distances: *mut f32,
    ) -> i32;
}

/// Cached Metal availability: -1=unchecked, 0=unavailable, 1=available
static METAL_STATUS: AtomicI32 = AtomicI32::new(-1);

/// Minimum n*dim product to justify GPU dispatch over CPU SIMD.
/// Metal command buffer dispatch has ~450us fixed overhead on Apple Silicon.
/// CPU NEON SIMD processes ~1 float-op/ns. Break-even is roughly n*dim >= 500K.
/// Set conservatively to ensure GPU is always faster when triggered.
/// Per-iteration DiskANN search (64-128 neighbors) won't reach this threshold;
/// it activates when multi-query batching aggregates enough work.
pub const MIN_GPU_WORK: usize = 524288;

/// Check if Metal GPU acceleration is available (cached after first call).
pub fn is_metal_available() -> bool {
    let status = METAL_STATUS.load(Ordering::Relaxed);
    if status >= 0 {
        return status == 1;
    }
    let avail = unsafe { diskann_metal_available() };
    METAL_STATUS.store(avail, Ordering::Relaxed);
    avail == 1
}

/// Compute batch distances using Metal GPU.
///
/// `candidates` must be `n * dim` contiguous floats.
/// `metric`: 0=L2, 1=InnerProduct.
/// `out` must have length >= n.
///
/// Returns true on success. Returns false if Metal is unavailable,
/// the batch is too small, or the GPU dispatch fails.
pub fn metal_batch_distances(
    query: &[f32],
    candidates: &[f32],
    n: usize,
    dim: usize,
    metric: u8,
    out: &mut [f32],
) -> bool {
    if n == 0 || dim == 0 {
        return true; // nothing to compute
    }
    if n * dim < MIN_GPU_WORK || !is_metal_available() {
        return false;
    }
    debug_assert_eq!(candidates.len(), n * dim);
    debug_assert!(out.len() >= n);

    let ret = unsafe {
        diskann_metal_batch_distances(
            query.as_ptr(),
            candidates.as_ptr(),
            n as i32,
            dim as i32,
            metric as i32,
            out.as_mut_ptr(),
        )
    };
    ret == 0
}
