#ifdef FAISS_AVAILABLE

#include "annsearch_extension.hpp"
#include "faiss_index_manager.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/scalar_function.hpp"

namespace duckdb {

/// Scalar function: faiss_add(name VARCHAR, vector FLOAT[]) -> BIGINT
/// Adds a single vector to the named index. Returns ntotal after add.
static void FaissAddScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vec = args.data[0];
	auto &vec_vec = args.data[1];

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<int64_t>(result);

	UnifiedVectorFormat name_format;
	name_vec.ToUnifiedFormat(args.size(), name_format);
	auto name_data = UnifiedVectorFormat::GetData<string_t>(name_format);

	UnifiedVectorFormat list_format;
	vec_vec.ToUnifiedFormat(args.size(), list_format);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_format);

	auto &child_vec = ListVector::GetEntry(vec_vec);
	UnifiedVectorFormat child_format;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(vec_vec), child_format);
	auto child_data = UnifiedVectorFormat::GetData<float>(child_format);

	for (idx_t i = 0; i < args.size(); i++) {
		auto name_idx = name_format.sel->get_index(i);
		auto list_idx = list_format.sel->get_index(i);

		if (!name_format.validity.RowIsValid(name_idx) || !list_format.validity.RowIsValid(list_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto name = name_data[name_idx].GetString();
		auto &entry = list_entries[list_idx];

		auto lock = IndexManager::Get().GetWrite(name);
		if (!lock) {
			throw InvalidInputException("Index '%s' not found", name);
		}

		int dim = lock->d;
		if ((int)entry.length != dim) {
			throw InvalidInputException("Expected vector of dimension %d, got %d", dim, (int)entry.length);
		}

		// Extract floats from child vector
		std::vector<float> vec(dim);
		for (idx_t j = 0; j < entry.length; j++) {
			auto child_idx = child_format.sel->get_index(entry.offset + j);
			vec[j] = child_data[child_idx];
		}

		lock->add(1, vec.data());
		result_data[i] = lock->ntotal;
	}
}

void RegisterFaissAddFunction(ExtensionLoader &loader) {
	auto func = ScalarFunction("faiss_add", {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT)},
	                           LogicalType::BIGINT, FaissAddScalarFun);
	func.stability = FunctionStability::VOLATILE;
	loader.RegisterFunction(func);
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
