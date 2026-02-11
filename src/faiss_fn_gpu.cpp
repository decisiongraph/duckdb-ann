#ifdef FAISS_AVAILABLE

#include "annsearch_extension.hpp"
#include "faiss_index_manager.hpp"
#include "gpu_backend.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

// ========================================
// faiss_to_gpu(name)
// ========================================

struct FaissToGpuBindData : public TableFunctionData {
	std::string name;
};

struct FaissToGpuState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissToGpuBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<FaissToGpuBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();
	names.push_back("status");
	return_types.push_back(LogicalType::VARCHAR);
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> FaissToGpuInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<FaissToGpuState>();
}

static void FaissToGpuScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<FaissToGpuBindData>();
	auto &state = data.global_state->Cast<FaissToGpuState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto &gpu = GetGpuBackend();
	if (!gpu.IsAvailable()) {
		throw InvalidInputException("No GPU backend available. %s", gpu.DeviceInfo());
	}

	auto lock = IndexManager::Get().GetWrite(bind_data.name);
	if (!lock) {
		throw InvalidInputException("Index '%s' not found", bind_data.name);
	}

	if (lock.managed->backend == "metal" || lock.managed->backend == "cuda") {
		throw InvalidInputException("Index '%s' is already on GPU (%s)", bind_data.name, lock.managed->backend);
	}

	auto gpu_index = gpu.CpuToGpu(lock.managed->index.get());
	std::string backend = "metal"; // TODO: detect from backend type
	IndexManager::Get().ReplaceIndex(lock.managed, std::move(gpu_index), backend);

	output.SetCardinality(1);
	output.data[0].SetValue(0, Value("Moved index '" + bind_data.name + "' to GPU (" + gpu.DeviceInfo() + ")"));
}

// ========================================
// faiss_to_cpu(name)
// ========================================

struct FaissToCpuBindData : public TableFunctionData {
	std::string name;
};

struct FaissToCpuState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissToCpuBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<FaissToCpuBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();
	names.push_back("status");
	return_types.push_back(LogicalType::VARCHAR);
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> FaissToCpuInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<FaissToCpuState>();
}

static void FaissToCpuScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<FaissToCpuBindData>();
	auto &state = data.global_state->Cast<FaissToCpuState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto lock = IndexManager::Get().GetWrite(bind_data.name);
	if (!lock) {
		throw InvalidInputException("Index '%s' not found", bind_data.name);
	}

	if (lock.managed->backend == "cpu") {
		throw InvalidInputException("Index '%s' is already on CPU", bind_data.name);
	}

	auto &gpu = GetGpuBackend();
	auto cpu_index = gpu.GpuToCpu(lock.managed->index.get());
	IndexManager::Get().ReplaceIndex(lock.managed, std::move(cpu_index), "cpu");

	output.SetCardinality(1);
	output.data[0].SetValue(0, Value("Moved index '" + bind_data.name + "' back to CPU"));
}

// ========================================
// faiss_gpu_info()
// ========================================

struct FaissGpuInfoState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissGpuInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	names.push_back("available");
	return_types.push_back(LogicalType::BOOLEAN);
	names.push_back("device");
	return_types.push_back(LogicalType::VARCHAR);
	return make_uniq<TableFunctionData>();
}

static unique_ptr<GlobalTableFunctionState> FaissGpuInfoInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<FaissGpuInfoState>();
}

static void FaissGpuInfoScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<FaissGpuInfoState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto &gpu = GetGpuBackend();
	output.SetCardinality(1);
	output.data[0].SetValue(0, Value::BOOLEAN(gpu.IsAvailable()));
	output.data[1].SetValue(0, Value(gpu.DeviceInfo()));
}

// ========================================
// Registration
// ========================================

void RegisterFaissGpuFunctions(ExtensionLoader &loader) {
	// faiss_to_gpu(name)
	{
		TableFunctionSet set("faiss_to_gpu");
		set.AddFunction(TableFunction({LogicalType::VARCHAR}, FaissToGpuScan, FaissToGpuBind, FaissToGpuInit));
		loader.RegisterFunction(set);
	}

	// faiss_to_cpu(name)
	{
		TableFunctionSet set("faiss_to_cpu");
		set.AddFunction(TableFunction({LogicalType::VARCHAR}, FaissToCpuScan, FaissToCpuBind, FaissToCpuInit));
		loader.RegisterFunction(set);
	}

	// faiss_gpu_info()
	{
		TableFunctionSet set("faiss_gpu_info");
		set.AddFunction(TableFunction({}, FaissGpuInfoScan, FaissGpuInfoBind, FaissGpuInfoInit));
		loader.RegisterFunction(set);
	}
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
