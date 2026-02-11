#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace duckdb {

// ========================================
// Rust FFI wrapper functions
// ========================================

// Index lifecycle
std::string DiskannCreateIndex(const std::string &name, int32_t dimension, const std::string &metric,
                               int32_t max_degree, int32_t build_complexity);

std::string DiskannDestroyIndex(const std::string &name);

// Vector operations
// Returns JSON: {"label": N}
std::string DiskannAddVector(const std::string &name, const float *vector, int32_t dimension);

// Returns JSON: {"results": [[label, distance], ...]}
std::string DiskannSearch(const std::string &name, const float *query, int32_t dimension, int32_t k);

// Management
// Returns JSON array of index info objects
std::string DiskannListIndexes();

// Returns JSON info object
std::string DiskannGetInfo(const std::string &name);

// Check availability
bool IsDiskannRustAvailable();
std::string GetDiskannRustVersion();

} // namespace duckdb
