#ifdef FAISS_AVAILABLE

#include "faiss_index_manager.hpp"
#include <faiss/MetricType.h>
#include <stdexcept>

namespace duckdb {

IndexManager &IndexManager::Get() {
	static IndexManager instance;
	return instance;
}

void IndexManager::Create(const std::string &name, std::unique_ptr<faiss::Index> index, const std::string &index_type) {
	std::lock_guard<std::mutex> guard(global_mutex_);
	if (indexes_.count(name)) {
		throw std::runtime_error("Index '" + name + "' already exists");
	}
	auto managed = std::make_unique<ManagedIndex>(std::move(index), index_type, "cpu");
	indexes_[name] = std::move(managed);
}

void IndexManager::Destroy(const std::string &name) {
	std::lock_guard<std::mutex> guard(global_mutex_);
	auto it = indexes_.find(name);
	if (it == indexes_.end()) {
		throw std::runtime_error("Index '" + name + "' not found");
	}
	// Acquire exclusive lock on the index before removing
	std::unique_lock<std::shared_mutex> idx_lock(it->second->rw_lock);
	idx_lock.unlock();
	indexes_.erase(it);
}

bool IndexManager::Exists(const std::string &name) const {
	std::lock_guard<std::mutex> guard(global_mutex_);
	return indexes_.count(name) > 0;
}

IndexManager::ReadLock IndexManager::GetRead(const std::string &name) {
	std::lock_guard<std::mutex> guard(global_mutex_);
	auto it = indexes_.find(name);
	if (it == indexes_.end()) {
		return ReadLock {nullptr, {}};
	}
	auto *managed = it->second.get();
	std::shared_lock<std::shared_mutex> lock(managed->rw_lock);
	return ReadLock {managed, std::move(lock)};
}

IndexManager::WriteLock IndexManager::GetWrite(const std::string &name) {
	std::lock_guard<std::mutex> guard(global_mutex_);
	auto it = indexes_.find(name);
	if (it == indexes_.end()) {
		return WriteLock {nullptr, {}};
	}
	auto *managed = it->second.get();
	std::unique_lock<std::shared_mutex> lock(managed->rw_lock);
	return WriteLock {managed, std::move(lock)};
}

void IndexManager::ReplaceIndex(ManagedIndex *managed, std::unique_ptr<faiss::Index> new_index,
                                const std::string &new_backend) {
	managed->index = std::move(new_index);
	managed->backend = new_backend;
}

static std::string MetricToString(faiss::MetricType m) {
	switch (m) {
	case faiss::METRIC_L2:
		return "L2";
	case faiss::METRIC_INNER_PRODUCT:
		return "IP";
	default:
		return "unknown";
	}
}

std::vector<IndexManager::IndexInfo> IndexManager::List() const {
	std::lock_guard<std::mutex> guard(global_mutex_);
	std::vector<IndexInfo> result;
	result.reserve(indexes_.size());
	for (auto &[name, managed] : indexes_) {
		std::shared_lock<std::shared_mutex> lock(managed->rw_lock);
		result.push_back({name, (int)managed->index->d, managed->index->ntotal,
		                  MetricToString(managed->index->metric_type), managed->index_type, managed->backend});
	}
	return result;
}

IndexManager::IndexInfo IndexManager::Info(const std::string &name) const {
	std::lock_guard<std::mutex> guard(global_mutex_);
	auto it = indexes_.find(name);
	if (it == indexes_.end()) {
		throw std::runtime_error("Index '" + name + "' not found");
	}
	auto &managed = it->second;
	std::shared_lock<std::shared_mutex> lock(managed->rw_lock);
	return {name,
	        (int)managed->index->d,
	        managed->index->ntotal,
	        MetricToString(managed->index->metric_type),
	        managed->index_type,
	        managed->backend};
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
