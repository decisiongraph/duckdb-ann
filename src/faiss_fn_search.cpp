#ifdef FAISS_AVAILABLE

#include "annsearch_extension.hpp"
#include "faiss_index_manager.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

// Helper declared in faiss_vector_utils.cpp
std::vector<float> FaissListToFloatVector(const Value &list_val, int expected_dim);

struct FaissSearchBindData : public TableFunctionData {
	std::string name;
	std::vector<float> query_vec;
	int k;
};

struct FaissSearchGlobalState : public GlobalTableFunctionState {
	std::vector<faiss::idx_t> labels;
	std::vector<float> distances;
	idx_t position = 0;
	idx_t result_count = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissSearchBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<FaissSearchBindData>();

	bind_data->name = input.inputs[0].GetValue<string>();

	// Extract query vector at bind time
	auto &vec_val = input.inputs[1];
	bind_data->query_vec = FaissListToFloatVector(vec_val, 0); // dimension checked at scan time

	bind_data->k = input.inputs[2].GetValue<int>();
	if (bind_data->k <= 0) {
		throw InvalidInputException("k must be positive, got %d", bind_data->k);
	}

	// Output: (label BIGINT, distance FLOAT)
	names.push_back("label");
	return_types.push_back(LogicalType::BIGINT);
	names.push_back("distance");
	return_types.push_back(LogicalType::FLOAT);

	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> FaissSearchInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<FaissSearchBindData>();
	auto state = make_uniq<FaissSearchGlobalState>();

	// Perform the search at init time (results are typically small)
	auto lock = IndexManager::Get().GetRead(bind_data.name);
	if (!lock) {
		throw InvalidInputException("Index '%s' not found", bind_data.name);
	}

	int dim = lock->d;
	if ((int)bind_data.query_vec.size() != dim) {
		throw InvalidInputException("Query vector dimension %d doesn't match index dimension %d",
		                            (int)bind_data.query_vec.size(), dim);
	}

	int k = bind_data.k;
	// Clamp k to ntotal if index has fewer vectors
	if (lock->ntotal < k) {
		k = (int)lock->ntotal;
	}
	if (k == 0) {
		state->result_count = 0;
		return std::move(state);
	}

	state->labels.resize(k);
	state->distances.resize(k);

	lock->search(1, bind_data.query_vec.data(), k, state->distances.data(), state->labels.data());

	// Count valid results (FAISS returns -1 for missing results)
	state->result_count = 0;
	for (int i = 0; i < k; i++) {
		if (state->labels[i] >= 0) {
			state->result_count++;
		} else {
			break;
		}
	}

	return std::move(state);
}

static void FaissSearchScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<FaissSearchGlobalState>();

	idx_t count = 0;
	while (state.position < state.result_count && count < STANDARD_VECTOR_SIZE) {
		output.data[0].SetValue(count, Value::BIGINT(state.labels[state.position]));
		output.data[1].SetValue(count, Value::FLOAT(state.distances[state.position]));
		state.position++;
		count++;
	}
	output.SetCardinality(count);
}

void RegisterFaissSearchFunction(ExtensionLoader &loader) {
	TableFunctionSet set("faiss_search");

	// faiss_search(name, query_vector, k) -> (label BIGINT, distance FLOAT)
	auto func = TableFunction({LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER},
	                          FaissSearchScan, FaissSearchBind, FaissSearchInit);
	set.AddFunction(func);
	loader.RegisterFunction(set);
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
