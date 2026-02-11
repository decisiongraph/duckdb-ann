#ifdef FAISS_AVAILABLE

#include "annsearch_extension.hpp"
#include "faiss_index_manager.hpp"
#include "duckdb/function/table_function.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/MetricType.h>
#include <faiss/index_factory.h>

namespace duckdb {

struct FaissCreateBindData : public TableFunctionData {
	std::string name;
	int dimension;
	std::string index_type;
	std::string metric;
	std::string description; // FAISS index_factory string (e.g., "PCA64,IVF4096,SQ8")
	// HNSW params
	int hnsw_m = 32;
	// IVF params
	int ivf_nlist = 100;
};

struct FaissCreateGlobalState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissCreateBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<FaissCreateBindData>();

	// Positional: name, dimension
	bind_data->name = input.inputs[0].GetValue<string>();
	bind_data->dimension = input.inputs[1].GetValue<int>();

	if (bind_data->dimension <= 0) {
		throw InvalidInputException("Dimension must be positive, got %d", bind_data->dimension);
	}

	// Named parameters
	for (auto &kv : input.named_parameters) {
		if (kv.first == "metric") {
			bind_data->metric = kv.second.GetValue<string>();
		} else if (kv.first == "type") {
			bind_data->index_type = kv.second.GetValue<string>();
		} else if (kv.first == "hnsw_m") {
			bind_data->hnsw_m = kv.second.GetValue<int>();
		} else if (kv.first == "ivf_nlist") {
			bind_data->ivf_nlist = kv.second.GetValue<int>();
		} else if (kv.first == "description") {
			bind_data->description = kv.second.GetValue<string>();
		}
	}

	// Defaults
	if (bind_data->index_type.empty()) {
		bind_data->index_type = "Flat";
	}
	if (bind_data->metric.empty()) {
		bind_data->metric = "L2";
	}

	names.push_back("status");
	return_types.push_back(LogicalType::VARCHAR);
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> FaissCreateInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<FaissCreateGlobalState>();
}

static void FaissCreateScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<FaissCreateBindData>();
	auto &state = data.global_state->Cast<FaissCreateGlobalState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	// Parse metric
	faiss::MetricType metric;
	if (bind_data.metric == "L2" || bind_data.metric == "l2") {
		metric = faiss::METRIC_L2;
	} else if (bind_data.metric == "IP" || bind_data.metric == "ip" || bind_data.metric == "inner_product") {
		metric = faiss::METRIC_INNER_PRODUCT;
	} else {
		throw InvalidInputException("Unknown metric '%s'. Supported: L2, IP", bind_data.metric);
	}

	// Create index -- prefer factory string if provided, otherwise use type shortcuts
	std::unique_ptr<faiss::Index> index;
	std::string type_name = bind_data.index_type;

	if (!bind_data.description.empty()) {
		// Use FAISS index_factory for complex index pipelines
		type_name = bind_data.description;
		index.reset(faiss::index_factory(bind_data.dimension, bind_data.description.c_str(), metric));
	} else if (type_name == "Flat" || type_name == "flat") {
		type_name = "Flat";
		index = std::make_unique<faiss::IndexFlat>(bind_data.dimension, metric);
	} else if (type_name == "HNSW" || type_name == "hnsw") {
		type_name = "HNSW";
		index = std::make_unique<faiss::IndexHNSWFlat>(bind_data.dimension, bind_data.hnsw_m, metric);
	} else if (type_name == "IVFFlat" || type_name == "ivfflat") {
		type_name = "IVFFlat";
		auto quantizer = new faiss::IndexFlat(bind_data.dimension, metric);
		index = std::make_unique<faiss::IndexIVFFlat>(quantizer, bind_data.dimension, bind_data.ivf_nlist, metric);
		// IndexIVFFlat takes ownership of quantizer
		static_cast<faiss::IndexIVFFlat *>(index.get())->own_fields = true;
	} else {
		throw InvalidInputException("Unknown index type '%s'. Use description := '...' for FAISS factory strings, "
		                            "or type := 'Flat'|'HNSW'|'IVFFlat'",
		                            type_name);
	}

	IndexManager::Get().Create(bind_data.name, std::move(index), type_name);

	output.SetCardinality(1);
	output.data[0].SetValue(0, Value("Created index '" + bind_data.name + "' (type=" + type_name + ", dim=" +
	                                 std::to_string(bind_data.dimension) + ", metric=" + bind_data.metric + ")"));
}

void RegisterFaissCreateFunction(ExtensionLoader &loader) {
	TableFunctionSet set("faiss_create");

	auto func =
	    TableFunction({LogicalType::VARCHAR, LogicalType::INTEGER}, FaissCreateScan, FaissCreateBind, FaissCreateInit);
	func.named_parameters["metric"] = LogicalType::VARCHAR;
	func.named_parameters["type"] = LogicalType::VARCHAR;
	func.named_parameters["description"] = LogicalType::VARCHAR; // FAISS index_factory string
	func.named_parameters["hnsw_m"] = LogicalType::INTEGER;
	func.named_parameters["ivf_nlist"] = LogicalType::INTEGER;

	set.AddFunction(func);
	loader.RegisterFunction(set);
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
