#include "annsearch_extension.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "rust_ffi.hpp"
#include "yyjson.hpp"

namespace duckdb {

using namespace duckdb_yyjson;

// ========================================
// Helper: parse DuckDB LIST<FLOAT> to float vector
// ========================================
static vector<float> ListToFloatVector(Vector &vec, idx_t row) {
	auto &list_entry = ListVector::GetData(vec)[row];
	auto &child = ListVector::GetEntry(vec);
	auto child_data = FlatVector::GetData<float>(child);
	vector<float> result;
	result.reserve(list_entry.length);
	for (idx_t i = 0; i < list_entry.length; i++) {
		result.push_back(child_data[list_entry.offset + i]);
	}
	return result;
}

// ========================================
// diskann_create(name, dimension, metric:='L2', max_degree:=64,
// build_complexity:=128)
// ========================================

struct DiskannCreateBindData : public TableFunctionData {
	string name;
	int32_t dimension;
	string metric;
	int32_t max_degree;
	int32_t build_complexity;
};

struct DiskannCreateState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> DiskannCreateBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<DiskannCreateBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();
	bind_data->dimension = input.inputs[1].GetValue<int32_t>();

	// Named parameters with defaults
	bind_data->metric = "L2";
	bind_data->max_degree = 64;
	bind_data->build_complexity = 128;

	for (auto &kv : input.named_parameters) {
		if (kv.first == "metric") {
			bind_data->metric = kv.second.GetValue<string>();
		} else if (kv.first == "max_degree") {
			bind_data->max_degree = kv.second.GetValue<int32_t>();
		} else if (kv.first == "build_complexity") {
			bind_data->build_complexity = kv.second.GetValue<int32_t>();
		}
	}

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("status");
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> DiskannCreateInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<DiskannCreateState>();
}

static void DiskannCreateScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<DiskannCreateState>();
	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto &bind = data.bind_data->Cast<DiskannCreateBindData>();
	auto result = DiskannCreateIndex(bind.name, bind.dimension, bind.metric, bind.max_degree, bind.build_complexity);

	// Parse status from JSON
	yyjson_doc *doc = yyjson_read(result.c_str(), result.length(), 0);
	string status = result;
	if (doc) {
		yyjson_val *root = yyjson_doc_get_root(doc);
		yyjson_val *status_val = yyjson_obj_get(root, "status");
		if (status_val && yyjson_is_str(status_val)) {
			status = yyjson_get_str(status_val);
		}
		yyjson_doc_free(doc);
	}

	output.SetCardinality(1);
	output.SetValue(0, 0, Value(status));
}

void RegisterDiskannCreateFunction(ExtensionLoader &loader) {
	TableFunction func("diskann_create", {LogicalType::VARCHAR, LogicalType::INTEGER}, DiskannCreateScan,
	                   DiskannCreateBind, DiskannCreateInit);
	func.named_parameters["metric"] = LogicalType::VARCHAR;
	func.named_parameters["max_degree"] = LogicalType::INTEGER;
	func.named_parameters["build_complexity"] = LogicalType::INTEGER;
	loader.RegisterFunction(func);
}

// ========================================
// diskann_destroy(name)
// ========================================

struct DiskannDestroyBindData : public TableFunctionData {
	string name;
};

struct DiskannDestroyState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> DiskannDestroyBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<DiskannDestroyBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("status");
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> DiskannDestroyInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<DiskannDestroyState>();
}

static void DiskannDestroyScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<DiskannDestroyState>();
	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto &bind = data.bind_data->Cast<DiskannDestroyBindData>();
	DiskannDestroyIndex(bind.name);
	output.SetCardinality(1);
	output.SetValue(0, 0, Value("Destroyed index '" + bind.name + "'"));
}

void RegisterDiskannDestroyFunction(ExtensionLoader &loader) {
	TableFunction func("diskann_destroy", {LogicalType::VARCHAR}, DiskannDestroyScan, DiskannDestroyBind,
	                   DiskannDestroyInit);
	loader.RegisterFunction(func);
}

// ========================================
// diskann_add(name, vector) -- scalar function
// Returns the label assigned to the vector
// ========================================

static void DiskannAddScalar(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vec = args.data[0];
	auto &list_vec = args.data[1];
	auto count = args.size();

	auto result_data = FlatVector::GetData<int64_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto name = name_vec.GetValue(i).GetValue<string>();
		auto floats = ListToFloatVector(list_vec, i);
		auto json = DiskannAddVector(name, floats.data(), static_cast<int32_t>(floats.size()));

		// Parse label from JSON {"label": N}
		int64_t label = -1;
		yyjson_doc *doc = yyjson_read(json.c_str(), json.length(), 0);
		if (doc) {
			yyjson_val *root = yyjson_doc_get_root(doc);
			yyjson_val *label_val = yyjson_obj_get(root, "label");
			if (label_val) {
				label = yyjson_get_int(label_val);
			}
			yyjson_doc_free(doc);
		}
		result_data[i] = label;
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

void RegisterDiskannAddFunction(ExtensionLoader &loader) {
	ScalarFunction func("diskann_add", {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT)},
	                    LogicalType::BIGINT, DiskannAddScalar);
	func.stability = FunctionStability::VOLATILE;
	loader.RegisterFunction(func);
}

// ========================================
// diskann_search(name, query_vec, k)
// Returns rows: (label BIGINT, distance FLOAT)
// ========================================

struct DiskannSearchBindData : public TableFunctionData {
	string name;
	vector<float> query;
	int32_t k;
};

struct DiskannSearchState : public GlobalTableFunctionState {
	vector<int64_t> labels;
	vector<float> distances;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> DiskannSearchBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<DiskannSearchBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();

	// Extract float list from second argument
	auto list_val = input.inputs[1];
	auto &children = ListValue::GetChildren(list_val);
	for (auto &child : children) {
		bind_data->query.push_back(child.GetValue<float>());
	}

	bind_data->k = input.inputs[2].GetValue<int32_t>();

	return_types.push_back(LogicalType::BIGINT);
	return_types.push_back(LogicalType::FLOAT);
	names.push_back("label");
	names.push_back("distance");
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> DiskannSearchInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<DiskannSearchState>();
	auto &bind = input.bind_data->Cast<DiskannSearchBindData>();

	// Perform search at init time
	auto json = DiskannSearch(bind.name, bind.query.data(), static_cast<int32_t>(bind.query.size()), bind.k);

	// Parse results from JSON: {"results": [[label, distance], ...]}
	yyjson_doc *doc = yyjson_read(json.c_str(), json.length(), 0);
	if (doc) {
		yyjson_val *root = yyjson_doc_get_root(doc);
		yyjson_val *results = yyjson_obj_get(root, "results");
		if (results && yyjson_is_arr(results)) {
			size_t idx, max_idx;
			yyjson_val *pair;
			yyjson_arr_foreach(results, idx, max_idx, pair) {
				if (yyjson_is_arr(pair) && yyjson_arr_size(pair) == 2) {
					yyjson_val *label = yyjson_arr_get(pair, 0);
					yyjson_val *dist = yyjson_arr_get(pair, 1);
					state->labels.push_back(yyjson_get_int(label));
					state->distances.push_back(static_cast<float>(yyjson_get_num(dist)));
				}
			}
		}
		yyjson_doc_free(doc);
	}

	return std::move(state);
}

static void DiskannSearchScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<DiskannSearchState>();

	if (state.position >= state.labels.size()) {
		output.SetCardinality(0);
		return;
	}

	idx_t chunk_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.labels.size() - state.position);

	auto label_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto dist_data = FlatVector::GetData<float>(output.data[1]);

	for (idx_t i = 0; i < chunk_size; i++) {
		label_data[i] = state.labels[state.position + i];
		dist_data[i] = state.distances[state.position + i];
	}

	state.position += chunk_size;
	output.SetCardinality(chunk_size);
}

void RegisterDiskannSearchFunction(ExtensionLoader &loader) {
	TableFunction func("diskann_search",
	                   {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER},
	                   DiskannSearchScan, DiskannSearchBind, DiskannSearchInit);
	loader.RegisterFunction(func);
}

// ========================================
// diskann_list()
// Returns: (name VARCHAR, dimension INTEGER, count BIGINT, metric VARCHAR,
// max_degree INTEGER)
// ========================================

struct DiskannListState : public GlobalTableFunctionState {
	vector<string> names;
	vector<int32_t> dimensions;
	vector<int64_t> counts;
	vector<string> metrics;
	vector<int32_t> degrees;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> DiskannListBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::INTEGER);
	return_types.push_back(LogicalType::BIGINT);
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::INTEGER);
	names.push_back("name");
	names.push_back("dimension");
	names.push_back("count");
	names.push_back("metric");
	names.push_back("max_degree");
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> DiskannListInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<DiskannListState>();
	auto json = DiskannListIndexes();

	yyjson_doc *doc = yyjson_read(json.c_str(), json.length(), 0);
	if (doc) {
		yyjson_val *root = yyjson_doc_get_root(doc);
		if (yyjson_is_arr(root)) {
			size_t idx, max_idx;
			yyjson_val *item;
			yyjson_arr_foreach(root, idx, max_idx, item) {
				yyjson_val *n = yyjson_obj_get(item, "name");
				yyjson_val *d = yyjson_obj_get(item, "dimension");
				yyjson_val *c = yyjson_obj_get(item, "count");
				yyjson_val *m = yyjson_obj_get(item, "metric");
				yyjson_val *g = yyjson_obj_get(item, "max_degree");
				state->names.push_back(n ? yyjson_get_str(n) : "");
				state->dimensions.push_back(d ? static_cast<int32_t>(yyjson_get_int(d)) : 0);
				state->counts.push_back(c ? yyjson_get_int(c) : 0);
				state->metrics.push_back(m ? yyjson_get_str(m) : "");
				state->degrees.push_back(g ? static_cast<int32_t>(yyjson_get_int(g)) : 0);
			}
		}
		yyjson_doc_free(doc);
	}

	return std::move(state);
}

static void DiskannListScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<DiskannListState>();

	if (state.position >= state.names.size()) {
		output.SetCardinality(0);
		return;
	}

	idx_t chunk_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.names.size() - state.position);

	for (idx_t i = 0; i < chunk_size; i++) {
		idx_t pos = state.position + i;
		output.SetValue(0, i, Value(state.names[pos]));
		output.SetValue(1, i, Value::INTEGER(state.dimensions[pos]));
		output.SetValue(2, i, Value::BIGINT(state.counts[pos]));
		output.SetValue(3, i, Value(state.metrics[pos]));
		output.SetValue(4, i, Value::INTEGER(state.degrees[pos]));
	}

	state.position += chunk_size;
	output.SetCardinality(chunk_size);
}

void RegisterDiskannListFunction(ExtensionLoader &loader) {
	TableFunction func("diskann_list", {}, DiskannListScan, DiskannListBind, DiskannListInit);
	loader.RegisterFunction(func);
}

// ========================================
// diskann_info(name)
// Returns: (key VARCHAR, value VARCHAR)
// ========================================

struct DiskannInfoBindData : public TableFunctionData {
	string name;
};

struct DiskannInfoState : public GlobalTableFunctionState {
	vector<string> keys;
	vector<string> values;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> DiskannInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<DiskannInfoBindData>();
	bind_data->name = input.inputs[0].GetValue<string>();
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("key");
	names.push_back("value");
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> DiskannInfoInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<DiskannInfoState>();
	auto &bind = input.bind_data->Cast<DiskannInfoBindData>();
	auto json = DiskannGetInfo(bind.name);

	yyjson_doc *doc = yyjson_read(json.c_str(), json.length(), 0);
	if (doc) {
		yyjson_val *root = yyjson_doc_get_root(doc);
		if (yyjson_is_obj(root)) {
			size_t idx, max_idx;
			yyjson_val *key, *val;
			yyjson_obj_foreach(root, idx, max_idx, key, val) {
				state->keys.push_back(yyjson_get_str(key));
				if (yyjson_is_str(val)) {
					state->values.push_back(yyjson_get_str(val));
				} else if (yyjson_is_int(val)) {
					state->values.push_back(std::to_string(yyjson_get_int(val)));
				} else {
					state->values.push_back(yyjson_get_str(val) ? yyjson_get_str(val) : "");
				}
			}
		}
		yyjson_doc_free(doc);
	}

	return std::move(state);
}

static void DiskannInfoScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<DiskannInfoState>();

	if (state.position >= state.keys.size()) {
		output.SetCardinality(0);
		return;
	}

	idx_t chunk_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.keys.size() - state.position);

	for (idx_t i = 0; i < chunk_size; i++) {
		idx_t pos = state.position + i;
		output.SetValue(0, i, Value(state.keys[pos]));
		output.SetValue(1, i, Value(state.values[pos]));
	}

	state.position += chunk_size;
	output.SetCardinality(chunk_size);
}

void RegisterDiskannInfoFunction(ExtensionLoader &loader) {
	TableFunction func("diskann_info", {LogicalType::VARCHAR}, DiskannInfoScan, DiskannInfoBind, DiskannInfoInit);
	loader.RegisterFunction(func);
}

} // namespace duckdb
