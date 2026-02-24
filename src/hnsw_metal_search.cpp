#ifdef FAISS_AVAILABLE
#if defined(FAISS_METAL_ENABLED) || defined(HAVE_ACCELERATE)

#include "hnsw_metal_search.hpp"
#include "metal_diskann_bridge.h"

// DuckDB redefines make_unique — include FAISS headers via wrapper
#include "faiss_wrapper.hpp"
#include <faiss/IndexHNSW.h>

#include <algorithm>
#include <cstring>
#include <queue>
#include <unordered_set>
#include <vector>

#ifdef HAVE_ACCELERATE
#include <vecLib/vDSP.h>
#endif

namespace duckdb {

/// GPU threshold — same as ann_search.cpp
static constexpr size_t MIN_GPU_WORK = 49152;

/// Compute batch distances: Metal GPU if large enough, else Accelerate/scalar CPU.
static void BatchDistances(const float *query, const float *candidates, int n, int dim, int metric, float *out) {
	auto work = static_cast<size_t>(n) * dim;

	// Try Metal GPU for large batches
	if (work >= MIN_GPU_WORK) {
		int rc = diskann_metal_batch_distances(query, candidates, n, dim, metric, out);
		if (rc == 0) {
			return;
		}
	}

#ifdef HAVE_ACCELERATE
	if (metric == 0) { // L2
		for (int i = 0; i < n; i++) {
			vDSP_distancesq(query, 1, candidates + i * dim, 1, &out[i], static_cast<vDSP_Length>(dim));
		}
	} else { // IP
		for (int i = 0; i < n; i++) {
			float dot;
			vDSP_dotpr(query, 1, candidates + i * dim, 1, &dot, static_cast<vDSP_Length>(dim));
			out[i] = -dot;
		}
	}
#else
	for (int i = 0; i < n; i++) {
		const float *cand = candidates + i * dim;
		float sum = 0;
		if (metric == 0) {
			for (int j = 0; j < dim; j++) {
				float d = query[j] - cand[j];
				sum += d * d;
			}
		} else {
			for (int j = 0; j < dim; j++) {
				sum += query[j] * cand[j];
			}
			sum = -sum;
		}
		out[i] = sum;
	}
#endif
}

/// Candidate entry for the search priority queues.
struct Candidate {
	float distance;
	int64_t id;

	bool operator>(const Candidate &o) const {
		return distance > o.distance;
	}
	bool operator<(const Candidate &o) const {
		return distance < o.distance;
	}
};

std::vector<std::pair<int64_t, float>> HnswMetalSearch(faiss::Index *index, const float *query, int32_t dimension,
                                                       int32_t k, int32_t ef_search,
                                                       const std::unordered_set<int64_t> &deleted_labels) {

	auto *hnsw_index = dynamic_cast<faiss::IndexHNSWFlat *>(index);
	if (!hnsw_index) {
		return {};
	}

	auto &hnsw = hnsw_index->hnsw;
	auto *storage = hnsw_index->storage;

	if (hnsw.entry_point < 0 || !storage || storage->ntotal == 0) {
		return {};
	}

	int metric = (index->metric_type == faiss::METRIC_INNER_PRODUCT) ? 1 : 0;
	int ef = (ef_search > 0) ? ef_search : hnsw.efSearch;
	if (ef < k) {
		ef = k;
	}

	std::unordered_set<int64_t> visited;
	visited.reserve(ef * 4);

	// Reconstruct entry point vector and compute initial distance
	std::vector<float> ep_vec(dimension);
	storage->reconstruct(hnsw.entry_point, ep_vec.data());

	float ep_dist;
	BatchDistances(query, ep_vec.data(), 1, dimension, metric, &ep_dist);

	int64_t ep = hnsw.entry_point;
	float ep_distance = ep_dist;

	// ---- Upper levels (greedy walk): levels max_level down to 1 ----
	for (int level = hnsw.max_level; level >= 1; level--) {
		bool changed = true;
		while (changed) {
			changed = false;

			size_t begin, end;
			hnsw.neighbor_range(ep, level, &begin, &end);

			// Collect valid neighbor IDs
			std::vector<int64_t> neighbor_ids;
			for (size_t j = begin; j < end; j++) {
				auto neighbor = hnsw.neighbors[j];
				if (neighbor < 0) {
					break;
				}
				neighbor_ids.push_back(neighbor);
			}

			if (neighbor_ids.empty()) {
				continue;
			}

			// Batch reconstruct
			std::vector<float> neighbor_vecs(neighbor_ids.size() * dimension);
			for (size_t i = 0; i < neighbor_ids.size(); i++) {
				storage->reconstruct(neighbor_ids[i], neighbor_vecs.data() + i * dimension);
			}

			// Batch distance computation
			std::vector<float> dists(neighbor_ids.size());
			BatchDistances(query, neighbor_vecs.data(), static_cast<int>(neighbor_ids.size()), dimension, metric,
			               dists.data());

			for (size_t i = 0; i < neighbor_ids.size(); i++) {
				if (dists[i] < ep_distance) {
					ep = neighbor_ids[i];
					ep_distance = dists[i];
					changed = true;
				}
			}
		}
	}

	// ---- Level 0: beam search with ef candidates ----
	// Min-heap: candidates to explore (nearest first)
	std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
	// Max-heap: top results (farthest first, so we can prune)
	std::priority_queue<Candidate> results;

	candidates.push({ep_distance, ep});
	results.push({ep_distance, ep});
	visited.insert(ep);

	while (!candidates.empty()) {
		auto current = candidates.top();
		candidates.pop();

		// If current candidate is farther than our worst result, we're done
		if (current.distance > results.top().distance && static_cast<int>(results.size()) >= ef) {
			break;
		}

		// Get level-0 neighbors
		size_t begin, end;
		hnsw.neighbor_range(current.id, 0, &begin, &end);

		// Collect unvisited neighbor IDs
		std::vector<int64_t> unvisited;
		for (size_t j = begin; j < end; j++) {
			auto neighbor = hnsw.neighbors[j];
			if (neighbor < 0) {
				break;
			}
			if (visited.count(neighbor) == 0) {
				visited.insert(neighbor);
				unvisited.push_back(neighbor);
			}
		}

		if (unvisited.empty()) {
			continue;
		}

		// Batch reconstruct neighbor vectors
		std::vector<float> neighbor_vecs(unvisited.size() * dimension);
		for (size_t i = 0; i < unvisited.size(); i++) {
			storage->reconstruct(unvisited[i], neighbor_vecs.data() + i * dimension);
		}

		// Batch distance computation (the hot path — Metal or Accelerate)
		std::vector<float> dists(unvisited.size());
		BatchDistances(query, neighbor_vecs.data(), static_cast<int>(unvisited.size()), dimension, metric,
		               dists.data());

		for (size_t i = 0; i < unvisited.size(); i++) {
			float d = dists[i];
			if (static_cast<int>(results.size()) < ef || d < results.top().distance) {
				candidates.push({d, unvisited[i]});
				results.push({d, unvisited[i]});
				if (static_cast<int>(results.size()) > ef) {
					results.pop(); // evict farthest
				}
			}
		}
	}

	// Extract results from max-heap, filter deleted, take top k
	std::vector<std::pair<int64_t, float>> output;
	output.reserve(results.size());
	while (!results.empty()) {
		auto &top = results.top();
		if (deleted_labels.count(top.id) == 0) {
			output.emplace_back(top.id, top.distance);
		}
		results.pop();
	}

	// Sort by distance ascending (heap gave us descending order)
	std::sort(output.begin(), output.end(), [](const std::pair<int64_t, float> &a, const std::pair<int64_t, float> &b) {
		return a.second < b.second;
	});

	// Trim to k
	if (static_cast<int>(output.size()) > k) {
		output.resize(k);
	}

	return output;
}

} // namespace duckdb

#endif // FAISS_METAL_ENABLED || HAVE_ACCELERATE
#endif // FAISS_AVAILABLE
