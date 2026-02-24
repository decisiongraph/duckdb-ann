#pragma once

#include <cstdint>

namespace duckdb {

// GPU acceleration mode for FAISS indexes (available without FAISS_AVAILABLE for optimizer)
enum class FaissGpuMode : uint8_t {
	CPU = 0, // never upload to GPU
	GPU = 1, // always upload, error if unavailable
	AUTO = 2 // heuristic decides after index build (default)
};

} // namespace duckdb

#ifdef FAISS_AVAILABLE

#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <faiss/Index.h>

#include <unordered_map>
#include <unordered_set>

namespace duckdb {

class DuckTableEntry;

// Shared FAISS option parsing — single source of truth
struct FaissParams {
	string metric = "L2";
	string index_type = "Flat";
	int32_t hnsw_m = 32;
	int32_t ivf_nlist = 100;
	int32_t nprobe = 1;
	int64_t train_sample = 0;
	string description;
	FaissGpuMode mode = FaissGpuMode::AUTO;

	static FaissParams Parse(const case_insensitive_map_t<Value> &options) {
		FaissParams p;
		bool has_mode = false;
		for (auto &kv : options) {
			if (kv.first == "metric") {
				p.metric = kv.second.ToString();
			} else if (kv.first == "type") {
				p.index_type = kv.second.ToString();
			} else if (kv.first == "hnsw_m") {
				p.hnsw_m = kv.second.GetValue<int32_t>();
			} else if (kv.first == "ivf_nlist") {
				p.ivf_nlist = kv.second.GetValue<int32_t>();
			} else if (kv.first == "nprobe") {
				p.nprobe = MaxValue<int32_t>(1, kv.second.GetValue<int32_t>());
			} else if (kv.first == "train_sample") {
				p.train_sample = kv.second.GetValue<int64_t>();
			} else if (kv.first == "description") {
				p.description = kv.second.ToString();
			} else if (kv.first == "mode") {
				has_mode = true;
				auto val = kv.second.ToString();
				if (val == "cpu" || val == "CPU") {
					p.mode = FaissGpuMode::CPU;
				} else if (val == "gpu" || val == "GPU") {
					p.mode = FaissGpuMode::GPU;
				} else if (val == "auto" || val == "AUTO") {
					p.mode = FaissGpuMode::AUTO;
				} else {
					throw InvalidInputException("Invalid mode '%s': expected 'cpu', 'gpu', or 'auto'", val);
				}
			} else if (kv.first == "gpu") {
				// Backward compat: gpu='true' -> GPU, gpu='false' -> CPU
				if (!has_mode) {
					bool gpu_val = BooleanValue::Get(kv.second.DefaultCastAs(LogicalType::BOOLEAN));
					p.mode = gpu_val ? FaissGpuMode::GPU : FaissGpuMode::CPU;
				}
			}
		}
		if (p.index_type.empty()) {
			p.index_type = "Flat";
		}
		return p;
	}

	case_insensitive_map_t<Value> ToOptions() const {
		case_insensitive_map_t<Value> opts;
		opts["metric"] = Value(metric);
		opts["type"] = Value(index_type);
		opts["hnsw_m"] = Value::INTEGER(hnsw_m);
		opts["ivf_nlist"] = Value::INTEGER(ivf_nlist);
		if (!description.empty()) {
			opts["description"] = Value(description);
		}
		switch (mode) {
		case FaissGpuMode::CPU:
			opts["mode"] = Value("cpu");
			break;
		case FaissGpuMode::GPU:
			opts["mode"] = Value("gpu");
			break;
		case FaissGpuMode::AUTO:
			opts["mode"] = Value("auto");
			break;
		}
		return opts;
	}
};

// ========================================
// FaissIndex: BoundIndex backed by libfaiss
// ========================================

class FaissIndex final : public BoundIndex {
public:
	static constexpr auto TYPE_NAME = "FAISS";

	FaissIndex(const string &name, IndexConstraintType constraint_type, const vector<column_t> &column_ids,
	           TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	           AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	           const IndexStorageInfo &info = IndexStorageInfo());

	~FaissIndex() override;

	// Static factories
	static unique_ptr<BoundIndex> Create(CreateIndexInput &input);
	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	// BoundIndex interface
	ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	void CommitDrop(IndexLock &lock) override;
	void Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;
	IndexStorageInfo SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) override;
	IndexStorageInfo SerializeToWAL(const case_insensitive_map_t<Value> &options) override;
	idx_t GetInMemorySize(IndexLock &state) override;
	bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;
	void Vacuum(IndexLock &state) override;
#ifdef DUCKDB_API_V15
	void Verify(IndexLock &state) override;
	string ToString(IndexLock &state, bool display_ascii = false) override;
#else
	string VerifyAndToString(IndexLock &state, const bool only_verify) override;
#endif
	void VerifyAllocations(IndexLock &state) override;
	void VerifyBuffers(IndexLock &l) override;
	string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
	                                     DataChunk &input) override;

	// ANN search
	vector<pair<row_t, float>> Search(const float *query, int32_t dimension, int32_t k);

	int32_t GetDimension() const {
		return dimension_;
	}
	const string &GetMetric() const {
		return metric_;
	}
	const string &GetFaissType() const {
		return index_type_;
	}
	int32_t GetNprobe() const {
		return nprobe_;
	}
	FaissGpuMode GetGpuMode() const {
		return mode_;
	}
	idx_t GetVectorCount() const {
		return faiss_index_ ? static_cast<idx_t>(faiss_index_->ntotal) : 0;
	}
	idx_t GetDeletedCount() const {
		return deleted_labels_.size();
	}

	friend class PhysicalCreateFaissIndex;

private:
	void PersistToDisk();
	void LoadFromStorage(const IndexStorageInfo &info);

	// FAISS index
	std::unique_ptr<faiss::Index> faiss_index_;

	// GPU acceleration helpers
	void EnsureGpuIndex();
	void InvalidateGpuIndex();

	// Index parameters
	int32_t dimension_ = 0;
	string metric_ = "L2";
	string index_type_ = "Flat";
	int32_t hnsw_m_ = 32;
	int32_t ivf_nlist_ = 100;
	int32_t nprobe_ = 1;
	int64_t train_sample_ = 0; // 0 = use all vectors for training
	string description_;
	FaissGpuMode mode_ = FaissGpuMode::AUTO;

	// GPU-resident copy of faiss_index_ (for search acceleration)
	std::unique_ptr<faiss::Index> gpu_index_;

	// Row ID mapping: internal label (0,1,2,...) <-> DuckDB row_t
	vector<row_t> label_to_rowid_;
	unordered_map<row_t, int64_t> rowid_to_label_;

	// Tombstones for deleted vectors
	unordered_set<int64_t> deleted_labels_;

	// Block storage for serialized data
	unique_ptr<FixedSizeAllocator> block_allocator_;
	IndexPointer root_block_ptr_;
	bool is_dirty_ = false;
};

// ========================================
// PhysicalCreateFaissIndex: CREATE INDEX operator
// ========================================

class PhysicalCreateFaissIndex : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::CREATE_INDEX;

	PhysicalCreateFaissIndex(PhysicalPlan &physical_plan, LogicalOperator &op, TableCatalogEntry &table,
	                         const vector<column_t> &column_ids, unique_ptr<CreateIndexInfo> info,
	                         vector<unique_ptr<Expression>> unbound_expressions, idx_t estimated_cardinality,
	                         unique_ptr<AlterTableInfo> alter_table_info = nullptr);

	DuckTableEntry &table;
	vector<column_t> storage_ids;
	unique_ptr<CreateIndexInfo> info;
	vector<unique_ptr<Expression>> unbound_expressions;
	unique_ptr<AlterTableInfo> alter_table_info;

public:
#ifdef DUCKDB_API_V15
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
#else
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;
#endif
	bool IsSource() const override {
		return true;
	}

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return false;
	}
};

// Function registration
void RegisterFaissIndexScanFunction(ExtensionLoader &loader);

} // namespace duckdb

#endif // FAISS_AVAILABLE
