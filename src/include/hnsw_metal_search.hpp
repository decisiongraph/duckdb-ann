#pragma once

#ifdef FAISS_AVAILABLE
#if defined(FAISS_METAL_ENABLED) || defined(HAVE_ACCELERATE)

#include <faiss/Index.h>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

namespace duckdb {

/// Metal/Accelerate-accelerated HNSW search using batch distance dispatch.
/// Traverses the HNSW graph via FAISS public API, batches neighbor vectors,
/// and dispatches distance computation to Metal GPU or Accelerate vDSP.
///
/// Returns up to k (label, distance) pairs sorted by distance.
std::vector<std::pair<int64_t, float>> HnswMetalSearch(faiss::Index *index, // Must be IndexHNSWFlat
                                                       const float *query, int32_t dimension, int32_t k,
                                                       int32_t ef_search, // 0 = use index default
                                                       const std::unordered_set<int64_t> &deleted_labels);

} // namespace duckdb

#endif // FAISS_METAL_ENABLED || HAVE_ACCELERATE
#endif // FAISS_AVAILABLE
