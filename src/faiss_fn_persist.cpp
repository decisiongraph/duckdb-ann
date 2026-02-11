#ifdef FAISS_AVAILABLE

#include "annsearch_extension.hpp"
#include "faiss_index_manager.hpp"
#include "duckdb/function/table_function.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>

namespace duckdb {

// ========================================
// faiss_save(name, path)
// ========================================

struct FaissSaveBindData : public TableFunctionData {
	std::string name;
	std::string path;
};

struct FaissSaveState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissSaveBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<FaissSaveBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();
	bind_data->path = input.inputs[1].GetValue<string>();
	names.push_back("status");
	return_types.push_back(LogicalType::VARCHAR);
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> FaissSaveInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<FaissSaveState>();
}

static void FaissSaveScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<FaissSaveBindData>();
	auto &state = data.global_state->Cast<FaissSaveState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto lock = IndexManager::Get().GetRead(bind_data.name);
	if (!lock) {
		throw InvalidInputException("Index '%s' not found", bind_data.name);
	}

	faiss::write_index(lock.managed->index.get(), bind_data.path.c_str());

	output.SetCardinality(1);
	output.data[0].SetValue(0, Value("Saved index '" + bind_data.name + "' to " + bind_data.path + " (" +
	                                 std::to_string(lock->ntotal) + " vectors)"));
}

// ========================================
// faiss_load(name, path)
// ========================================

struct FaissLoadBindData : public TableFunctionData {
	std::string name;
	std::string path;
};

struct FaissLoadState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissLoadBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<FaissLoadBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();
	bind_data->path = input.inputs[1].GetValue<string>();
	names.push_back("status");
	return_types.push_back(LogicalType::VARCHAR);
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> FaissLoadInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<FaissLoadState>();
}

static void FaissLoadScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<FaissLoadBindData>();
	auto &state = data.global_state->Cast<FaissLoadState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	std::unique_ptr<faiss::Index> index(faiss::read_index(bind_data.path.c_str()));
	if (!index) {
		throw InvalidInputException("Failed to load index from '%s'", bind_data.path);
	}

	auto ntotal = index->ntotal;
	auto dim = index->d;

	// Detect index type from the loaded index
	std::string index_type = "Unknown";
	if (dynamic_cast<faiss::IndexFlat *>(index.get())) {
		index_type = "Flat";
	} else if (dynamic_cast<faiss::IndexHNSW *>(index.get())) {
		index_type = "HNSW";
	} else if (dynamic_cast<faiss::IndexIVF *>(index.get())) {
		index_type = "IVF";
	}

	IndexManager::Get().Create(bind_data.name, std::move(index), index_type);

	output.SetCardinality(1);
	output.data[0].SetValue(0, Value("Loaded index '" + bind_data.name + "' from " + bind_data.path +
	                                 " (dim=" + std::to_string(dim) + ", vectors=" + std::to_string(ntotal) + ")"));
}

// ========================================
// Registration
// ========================================

void RegisterFaissPersistFunctions(ExtensionLoader &loader) {
	// faiss_save(name, path)
	{
		TableFunctionSet set("faiss_save");
		set.AddFunction(
		    TableFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, FaissSaveScan, FaissSaveBind, FaissSaveInit));
		loader.RegisterFunction(set);
	}

	// faiss_load(name, path)
	{
		TableFunctionSet set("faiss_load");
		set.AddFunction(
		    TableFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, FaissLoadScan, FaissLoadBind, FaissLoadInit));
		loader.RegisterFunction(set);
	}
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
