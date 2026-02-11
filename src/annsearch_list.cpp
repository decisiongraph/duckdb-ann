#include "annsearch_extension.hpp"
#include "rust_ffi.hpp"
#include "yyjson.hpp"

#ifdef FAISS_AVAILABLE
#include "faiss_index_manager.hpp"
#endif

namespace duckdb {

using namespace duckdb_yyjson;

// ========================================
// annsearch_list()
// Unified listing combining DiskANN + FAISS indexes
// Returns: (name, engine, dimension, count, metric, type, backend)
// ========================================

struct AnnsearchListEntry {
	string name;
	string engine;
	int32_t dimension;
	int64_t count;
	string metric;
	string type;
	string backend;
};

struct AnnsearchListState : public GlobalTableFunctionState {
	vector<AnnsearchListEntry> entries;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> AnnsearchListBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::INTEGER);
	return_types.push_back(LogicalType::BIGINT);
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("name");
	names.push_back("engine");
	names.push_back("dimension");
	names.push_back("count");
	names.push_back("metric");
	names.push_back("type");
	names.push_back("backend");
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> AnnsearchListInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<AnnsearchListState>();

	// Collect DiskANN indexes via Rust FFI JSON
	auto json = DiskannListIndexes();
	yyjson_doc *doc = yyjson_read(json.c_str(), json.length(), 0);
	if (doc) {
		yyjson_val *root = yyjson_doc_get_root(doc);
		if (yyjson_is_arr(root)) {
			size_t idx, max_idx;
			yyjson_val *item;
			yyjson_arr_foreach(root, idx, max_idx, item) {
				AnnsearchListEntry entry;
				yyjson_val *n = yyjson_obj_get(item, "name");
				yyjson_val *d = yyjson_obj_get(item, "dimension");
				yyjson_val *c = yyjson_obj_get(item, "count");
				yyjson_val *m = yyjson_obj_get(item, "metric");
				entry.name = n ? yyjson_get_str(n) : "";
				entry.engine = "diskann";
				entry.dimension = d ? static_cast<int32_t>(yyjson_get_int(d)) : 0;
				entry.count = c ? yyjson_get_int(c) : 0;
				entry.metric = m ? yyjson_get_str(m) : "";
				entry.type = "vamana";
				entry.backend = "rust";
				state->entries.push_back(std::move(entry));
			}
		}
		yyjson_doc_free(doc);
	}

#ifdef FAISS_AVAILABLE
	// Collect FAISS indexes via IndexManager
	auto faiss_indexes = IndexManager::Get().List();
	for (auto &info : faiss_indexes) {
		AnnsearchListEntry entry;
		entry.name = info.name;
		entry.engine = "faiss";
		entry.dimension = info.dimension;
		entry.count = info.ntotal;
		entry.metric = info.metric;
		entry.type = info.index_type;
		entry.backend = info.backend;
		state->entries.push_back(std::move(entry));
	}
#endif

	return std::move(state);
}

static void AnnsearchListScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<AnnsearchListState>();

	if (state.position >= state.entries.size()) {
		output.SetCardinality(0);
		return;
	}

	idx_t chunk_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.entries.size() - state.position);

	for (idx_t i = 0; i < chunk_size; i++) {
		auto &entry = state.entries[state.position + i];
		output.SetValue(0, i, Value(entry.name));
		output.SetValue(1, i, Value(entry.engine));
		output.SetValue(2, i, Value::INTEGER(entry.dimension));
		output.SetValue(3, i, Value::BIGINT(entry.count));
		output.SetValue(4, i, Value(entry.metric));
		output.SetValue(5, i, Value(entry.type));
		output.SetValue(6, i, Value(entry.backend));
	}

	state.position += chunk_size;
	output.SetCardinality(chunk_size);
}

void RegisterAnnsearchListFunction(ExtensionLoader &loader) {
	TableFunction func("annsearch_list", {}, AnnsearchListScan, AnnsearchListBind, AnnsearchListInit);
	loader.RegisterFunction(func);
}

} // namespace duckdb
