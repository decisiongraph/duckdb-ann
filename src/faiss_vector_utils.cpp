#ifdef FAISS_AVAILABLE

#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include <vector>

namespace duckdb {

/// Extract a float vector from a DuckDB LIST<FLOAT> value.
/// Validates dimension matches expected_dim if > 0.
std::vector<float> FaissListToFloatVector(const Value &list_val, int expected_dim) {
	if (list_val.IsNull()) {
		throw InvalidInputException("Vector cannot be NULL");
	}
	auto &children = ListValue::GetChildren(list_val);
	if (expected_dim > 0 && (int)children.size() != expected_dim) {
		throw InvalidInputException("Expected vector of dimension %d, got %d", expected_dim, (int)children.size());
	}
	std::vector<float> result;
	result.reserve(children.size());
	for (auto &child : children) {
		result.push_back(child.GetValue<float>());
	}
	return result;
}

/// Convert a float array to a DuckDB LIST<FLOAT> value.
Value FaissFloatArrayToList(const float *data, int dim) {
	vector<Value> children;
	children.reserve(dim);
	for (int i = 0; i < dim; i++) {
		children.push_back(Value::FLOAT(data[i]));
	}
	return Value::LIST(LogicalType::FLOAT, std::move(children));
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
