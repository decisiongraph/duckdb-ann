#pragma once

#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/storage/data_table.hpp"
#include "rust_ffi.hpp"

#include <unordered_map>
#include <unordered_set>

namespace duckdb {

class DuckTableEntry;

// ========================================
// DiskannIndex: BoundIndex implementation
// ========================================

class DiskannIndex final : public BoundIndex {
public:
	static constexpr auto TYPE_NAME = "DISKANN";

	DiskannIndex(const string &name, IndexConstraintType constraint_type, const vector<column_t> &column_ids,
	             TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	             AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	             const IndexStorageInfo &info = IndexStorageInfo());

	~DiskannIndex() override;

	// Static factory for create_instance
	static unique_ptr<BoundIndex> Create(CreateIndexInput &input);

	// Static factory for create_plan
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
	string VerifyAndToString(IndexLock &state, const bool only_verify) override;
	void VerifyAllocations(IndexLock &state) override;
	void VerifyBuffers(IndexLock &l) override;
	string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
	                                     DataChunk &input) override;

	// ANN search (called by optimizer/index scan)
	vector<pair<row_t, float>> Search(const float *query, int32_t dimension, int32_t k, int32_t search_complexity);

	int32_t GetDimension() const {
		return dimension_;
	}
	idx_t GetVectorCount() const {
		return rust_handle_ ? static_cast<idx_t>(DiskannDetachedCount(rust_handle_)) : 0;
	}
	idx_t GetDeletedCount() const {
		return deleted_labels_.size();
	}

	// PhysicalCreateDiskannIndex needs to set internal state after build
	friend class PhysicalCreateDiskannIndex;

private:
	void PersistToDisk();
	void LoadFromStorage(const IndexStorageInfo &info);

	// Rust DiskANN index handle
	DiskannHandle rust_handle_ = nullptr;

	// Index parameters
	int32_t dimension_ = 0;
	string metric_ = "L2";
	int32_t max_degree_ = 64;
	int32_t build_complexity_ = 128;
	float alpha_ = 1.2f;

	// Row ID mapping: internal label (0,1,2,...) <-> DuckDB row_t
	vector<row_t> label_to_rowid_;
	unordered_map<row_t, uint32_t> rowid_to_label_;

	// Tombstones for deleted vectors
	unordered_set<uint32_t> deleted_labels_;

	// Block storage for serialized data
	unique_ptr<FixedSizeAllocator> block_allocator_;
	IndexPointer root_block_ptr_;
	bool is_dirty_ = false;
};

// ========================================
// PhysicalCreateDiskannIndex: CREATE INDEX operator
// ========================================

class PhysicalCreateDiskannIndex : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::CREATE_INDEX;

	PhysicalCreateDiskannIndex(PhysicalPlan &physical_plan, LogicalOperator &op, TableCatalogEntry &table,
	                           const vector<column_t> &column_ids, unique_ptr<CreateIndexInfo> info,
	                           vector<unique_ptr<Expression>> unbound_expressions, idx_t estimated_cardinality,
	                           unique_ptr<AlterTableInfo> alter_table_info = nullptr);

	DuckTableEntry &table;
	vector<column_t> storage_ids;
	unique_ptr<CreateIndexInfo> info;
	vector<unique_ptr<Expression>> unbound_expressions;
	unique_ptr<AlterTableInfo> alter_table_info;

public:
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;
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
		return false; // DiskANN insert is not thread-safe currently
	}
};

} // namespace duckdb
