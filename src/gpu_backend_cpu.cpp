#ifdef FAISS_AVAILABLE

#include "gpu_backend.hpp"
#include <stdexcept>

namespace duckdb {

/// CPU fallback: no GPU available.
class CpuGpuBackend : public GpuBackend {
public:
	bool IsAvailable() const override {
		return false;
	}

	std::string DeviceInfo() const override {
		return "No GPU backend available";
	}

	std::unique_ptr<faiss::Index> CpuToGpu(faiss::Index * /*cpu_index*/) override {
		throw std::runtime_error("No GPU backend available. Build with Metal (macOS) or CUDA (Linux/Windows) support.");
	}

	std::unique_ptr<faiss::Index> GpuToCpu(faiss::Index * /*gpu_index*/) override {
		throw std::runtime_error("No GPU backend available.");
	}
};

#ifndef FAISS_METAL_ENABLED
// When no GPU backend is compiled, return CPU fallback
GpuBackend &GetGpuBackend() {
	static CpuGpuBackend instance;
	return instance;
}
#endif

} // namespace duckdb

#endif // FAISS_AVAILABLE
