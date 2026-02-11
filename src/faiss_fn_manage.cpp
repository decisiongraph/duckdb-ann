#ifdef FAISS_AVAILABLE

#include "annsearch_extension.hpp"
#include "faiss_index_manager.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

// ========================================
// faiss_destroy(name)
// ========================================

struct FaissDestroyBindData : public TableFunctionData {
	std::string name;
};

struct FaissDestroyState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissDestroyBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<FaissDestroyBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();
	names.push_back("status");
	return_types.push_back(LogicalType::VARCHAR);
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> FaissDestroyInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<FaissDestroyState>();
}

static void FaissDestroyScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<FaissDestroyBindData>();
	auto &state = data.global_state->Cast<FaissDestroyState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	IndexManager::Get().Destroy(bind_data.name);

	output.SetCardinality(1);
	output.data[0].SetValue(0, Value("Destroyed index '" + bind_data.name + "'"));
}

// ========================================
// faiss_list()
// ========================================

struct FaissListState : public GlobalTableFunctionState {
	std::vector<IndexManager::IndexInfo> indexes;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissListBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	names.push_back("name");
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("dimension");
	return_types.push_back(LogicalType::INTEGER);
	names.push_back("count");
	return_types.push_back(LogicalType::BIGINT);
	names.push_back("metric");
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("type");
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("backend");
	return_types.push_back(LogicalType::VARCHAR);
	return make_uniq<TableFunctionData>();
}

static unique_ptr<GlobalTableFunctionState> FaissListInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<FaissListState>();
	state->indexes = IndexManager::Get().List();
	return std::move(state);
}

static void FaissListScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<FaissListState>();

	idx_t count = 0;
	while (state.position < state.indexes.size() && count < STANDARD_VECTOR_SIZE) {
		auto &info = state.indexes[state.position];
		output.data[0].SetValue(count, Value(info.name));
		output.data[1].SetValue(count, Value::INTEGER(info.dimension));
		output.data[2].SetValue(count, Value::BIGINT(info.ntotal));
		output.data[3].SetValue(count, Value(info.metric));
		output.data[4].SetValue(count, Value(info.index_type));
		output.data[5].SetValue(count, Value(info.backend));
		state.position++;
		count++;
	}
	output.SetCardinality(count);
}

// ========================================
// faiss_info(name)
// ========================================

struct FaissInfoBindData : public TableFunctionData {
	std::string name;
};

struct FaissInfoState : public GlobalTableFunctionState {
	std::vector<std::pair<std::string, std::string>> kvs;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<FaissInfoBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();
	names.push_back("key");
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("value");
	return_types.push_back(LogicalType::VARCHAR);
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> FaissInfoInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<FaissInfoBindData>();
	auto state = make_uniq<FaissInfoState>();

	auto info = IndexManager::Get().Info(bind_data.name);
	state->kvs.push_back({"name", info.name});
	state->kvs.push_back({"dimension", std::to_string(info.dimension)});
	state->kvs.push_back({"count", std::to_string(info.ntotal)});
	state->kvs.push_back({"metric", info.metric});
	state->kvs.push_back({"type", info.index_type});
	state->kvs.push_back({"backend", info.backend});

	return std::move(state);
}

static void FaissInfoScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<FaissInfoState>();

	idx_t count = 0;
	while (state.position < state.kvs.size() && count < STANDARD_VECTOR_SIZE) {
		output.data[0].SetValue(count, Value(state.kvs[state.position].first));
		output.data[1].SetValue(count, Value(state.kvs[state.position].second));
		state.position++;
		count++;
	}
	output.SetCardinality(count);
}

// ========================================
// Registration
// ========================================

void RegisterFaissManageFunctions(ExtensionLoader &loader) {
	// faiss_destroy(name)
	{
		TableFunctionSet set("faiss_destroy");
		set.AddFunction(TableFunction({LogicalType::VARCHAR}, FaissDestroyScan, FaissDestroyBind, FaissDestroyInit));
		loader.RegisterFunction(set);
	}

	// faiss_list()
	{
		TableFunctionSet set("faiss_list");
		set.AddFunction(TableFunction({}, FaissListScan, FaissListBind, FaissListInit));
		loader.RegisterFunction(set);
	}

	// faiss_info(name)
	{
		TableFunctionSet set("faiss_info");
		set.AddFunction(TableFunction({LogicalType::VARCHAR}, FaissInfoScan, FaissInfoBind, FaissInfoInit));
		loader.RegisterFunction(set);
	}
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
