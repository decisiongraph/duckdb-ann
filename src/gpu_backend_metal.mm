#ifdef FAISS_AVAILABLE

#include "gpu_backend.hpp"

#ifdef FAISS_METAL_ENABLED

#include <faiss-metal/MetalIndexFlat.h>
#include <faiss-metal/StandardMetalResources.h>
#include <faiss/IndexFlat.h>

namespace duckdb {

class MetalGpuBackend : public GpuBackend {
  public:
    MetalGpuBackend() {
        try {
            resources_ = std::make_shared<faiss_metal::StandardMetalResources>();
            available_ = true;
        } catch (...) {
            available_ = false;
        }
    }

    bool IsAvailable() const override {
        return available_;
    }

    std::string DeviceInfo() const override {
        if (!available_) {
            return "Metal: not available";
        }
        auto &caps = resources_->getCapabilities();
        return "Metal GPU (family=" + std::to_string(caps.metalFamily) + ")";
    }

    std::unique_ptr<faiss::Index> CpuToGpu(faiss::Index *cpu_index) override {
        if (!available_) {
            throw std::runtime_error("Metal GPU backend not available");
        }

        auto *flat = dynamic_cast<faiss::IndexFlat *>(cpu_index);
        if (!flat) {
            throw std::runtime_error("Metal GPU currently only supports IndexFlat. "
                                     "Got a non-Flat index type.");
        }

        return faiss_metal::index_cpu_to_metal(resources_, flat);
    }

    std::unique_ptr<faiss::Index> GpuToCpu(faiss::Index *gpu_index) override {
        auto *metal_flat = dynamic_cast<faiss_metal::MetalIndexFlat *>(gpu_index);
        if (!metal_flat) {
            throw std::runtime_error("Index is not a MetalIndexFlat -- cannot convert to CPU");
        }

        return faiss_metal::index_metal_to_cpu(metal_flat);
    }

  private:
    std::shared_ptr<faiss_metal::MetalResources> resources_;
    bool available_ = false;
};

GpuBackend &GetGpuBackend() {
    static MetalGpuBackend instance;
    return instance;
}

} // namespace duckdb

#endif // FAISS_METAL_ENABLED

#endif // FAISS_AVAILABLE
