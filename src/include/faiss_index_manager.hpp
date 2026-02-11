#pragma once

#ifdef FAISS_AVAILABLE

#include <faiss/Index.h>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

struct ManagedIndex {
	std::unique_ptr<faiss::Index> index;
	std::string index_type; // "Flat", "IVFFlat", "HNSW", etc.
	std::string backend;    // "cpu" or "metal"
	mutable std::shared_mutex rw_lock;

	ManagedIndex() = default;
	ManagedIndex(std::unique_ptr<faiss::Index> idx, std::string type, std::string bk)
	    : index(std::move(idx)), index_type(std::move(type)), backend(std::move(bk)) {
	}
};

/// Singleton registry of named FAISS indexes.
/// Thread-safe: global mutex for create/destroy, per-index shared_mutex for read/write.
class IndexManager {
public:
	static IndexManager &Get();

	/// Create a new named index. Throws if name already exists.
	void Create(const std::string &name, std::unique_ptr<faiss::Index> index, const std::string &index_type);

	/// Destroy a named index. Throws if not found.
	void Destroy(const std::string &name);

	/// Check if an index exists.
	bool Exists(const std::string &name) const;

	/// Get index with shared (read) lock. Caller holds lock via returned guard.
	/// Returns nullptr if not found.
	struct ReadLock {
		ManagedIndex *managed;
		std::shared_lock<std::shared_mutex> lock;
		faiss::Index *operator->() const {
			return managed->index.get();
		}
		faiss::Index &operator*() const {
			return *managed->index;
		}
		explicit operator bool() const {
			return managed != nullptr;
		}
	};
	ReadLock GetRead(const std::string &name);

	/// Get index with exclusive (write) lock.
	struct WriteLock {
		ManagedIndex *managed;
		std::unique_lock<std::shared_mutex> lock;
		faiss::Index *operator->() const {
			return managed->index.get();
		}
		faiss::Index &operator*() const {
			return *managed->index;
		}
		explicit operator bool() const {
			return managed != nullptr;
		}
	};
	WriteLock GetWrite(const std::string &name);

	/// Replace the index pointer (e.g., after CPU->GPU conversion). Caller must hold WriteLock.
	void ReplaceIndex(ManagedIndex *managed, std::unique_ptr<faiss::Index> new_index, const std::string &new_backend);

	/// List all index names with basic info.
	struct IndexInfo {
		std::string name;
		int dimension;
		int64_t ntotal;
		std::string metric;
		std::string index_type;
		std::string backend;
	};
	std::vector<IndexInfo> List() const;

	/// Get info for a single index by name. Throws if not found.
	IndexInfo Info(const std::string &name) const;

private:
	IndexManager() = default;
	mutable std::mutex global_mutex_;
	std::unordered_map<std::string, std::unique_ptr<ManagedIndex>> indexes_;
};

} // namespace duckdb

#endif // FAISS_AVAILABLE
