// Rust DiskANN FFI wrapper for DuckDB extension

#include "rust_ffi.hpp"
#include <string>

// Rust FFI declarations
extern "C" {
struct DiskannResult {
	char *json_ptr;
	char *error_ptr;
};

DiskannResult diskann_create_index(const char *name, int32_t dimension, const char *metric, int32_t max_degree,
                                   int32_t build_complexity);
DiskannResult diskann_destroy_index(const char *name);
DiskannResult diskann_add_vector(const char *name, const float *vector_ptr, int32_t dimension);
DiskannResult diskann_search(const char *name, const float *query_ptr, int32_t dimension, int32_t k);
DiskannResult diskann_list_indexes();
DiskannResult diskann_get_info(const char *name);
void diskann_free_result(DiskannResult result);
const char *diskann_rust_version();
}

namespace duckdb {

// RAII wrapper for DiskannResult
class RustResult {
public:
	explicit RustResult(DiskannResult result) : result_(result) {
	}
	~RustResult() {
		diskann_free_result(result_);
	}

	bool HasError() const {
		return result_.error_ptr != nullptr;
	}
	std::string GetError() const {
		return result_.error_ptr ? std::string(result_.error_ptr) : "";
	}
	std::string GetJson() const {
		return result_.json_ptr ? std::string(result_.json_ptr) : "{}";
	}

private:
	DiskannResult result_;
};

std::string DiskannCreateIndex(const std::string &name, int32_t dimension, const std::string &metric,
                               int32_t max_degree, int32_t build_complexity) {
	RustResult result(diskann_create_index(name.c_str(), dimension, metric.c_str(), max_degree, build_complexity));
	if (result.HasError()) {
		throw std::runtime_error("DiskANN create failed: " + result.GetError());
	}
	return result.GetJson();
}

std::string DiskannDestroyIndex(const std::string &name) {
	RustResult result(diskann_destroy_index(name.c_str()));
	if (result.HasError()) {
		throw std::runtime_error("DiskANN destroy failed: " + result.GetError());
	}
	return result.GetJson();
}

std::string DiskannAddVector(const std::string &name, const float *vector, int32_t dimension) {
	RustResult result(diskann_add_vector(name.c_str(), vector, dimension));
	if (result.HasError()) {
		throw std::runtime_error("DiskANN add failed: " + result.GetError());
	}
	return result.GetJson();
}

std::string DiskannSearch(const std::string &name, const float *query, int32_t dimension, int32_t k) {
	RustResult result(diskann_search(name.c_str(), query, dimension, k));
	if (result.HasError()) {
		throw std::runtime_error("DiskANN search failed: " + result.GetError());
	}
	return result.GetJson();
}

std::string DiskannListIndexes() {
	RustResult result(diskann_list_indexes());
	if (result.HasError()) {
		throw std::runtime_error("DiskANN list failed: " + result.GetError());
	}
	return result.GetJson();
}

std::string DiskannGetInfo(const std::string &name) {
	RustResult result(diskann_get_info(name.c_str()));
	if (result.HasError()) {
		throw std::runtime_error("DiskANN info failed: " + result.GetError());
	}
	return result.GetJson();
}

bool IsDiskannRustAvailable() {
	return true;
}

std::string GetDiskannRustVersion() {
	const char *ver = diskann_rust_version();
	return ver ? std::string(ver) : "unknown";
}

} // namespace duckdb
