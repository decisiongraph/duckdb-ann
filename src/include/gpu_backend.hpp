#pragma once

#ifdef FAISS_AVAILABLE

#include <faiss/Index.h>
#include <memory>
#include <string>

namespace duckdb {

/// Abstract GPU backend interface for FAISS indexes.
class GpuBackend {
public:
	virtual ~GpuBackend() = default;

	/// Whether this GPU backend is available on the current system.
	virtual bool IsAvailable() const = 0;

	/// Human-readable device info (e.g., "Apple M2 Pro (Metal)")
	virtual std::string DeviceInfo() const = 0;

	/// Backend name for index tracking (e.g., "metal", "cuda")
	virtual std::string BackendName() const = 0;

	/// Move a CPU index to GPU. Returns new GPU index. Throws on failure.
	virtual std::unique_ptr<faiss::Index> CpuToGpu(faiss::Index *cpu_index) = 0;

	/// Move a GPU index back to CPU. Returns new CPU index. Throws on failure.
	virtual std::unique_ptr<faiss::Index> GpuToCpu(faiss::Index *gpu_index) = 0;
};

/// Get the singleton GPU backend (Metal on macOS, CPU fallback otherwise).
GpuBackend &GetGpuBackend();

} // namespace duckdb

#endif // FAISS_AVAILABLE
