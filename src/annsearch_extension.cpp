#define DUCKDB_EXTENSION_MAIN

#include "annsearch_extension.hpp"
#include "duckdb.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	// DiskANN functions (always available)
	RegisterDiskannCreateFunction(loader);
	RegisterDiskannDestroyFunction(loader);
	RegisterDiskannAddFunction(loader);
	RegisterDiskannSearchFunction(loader);
	RegisterDiskannListFunction(loader);
	RegisterDiskannInfoFunction(loader);

#ifdef FAISS_AVAILABLE
	// FAISS functions (conditional on libfaiss)
	RegisterFaissCreateFunction(loader);
	RegisterFaissAddFunction(loader);
	RegisterFaissSearchFunction(loader);
	RegisterFaissPersistFunctions(loader);
	RegisterFaissManageFunctions(loader);
#ifdef FAISS_METAL_ENABLED
	RegisterFaissGpuFunctions(loader);
#endif
#endif

	// Unified listing (always available)
	RegisterAnnsearchListFunction(loader);
}

void AnnsearchExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string AnnsearchExtension::Name() {
	return "annsearch";
}

std::string AnnsearchExtension::Version() const {
#ifdef EXT_VERSION_ANNSEARCH
	return EXT_VERSION_ANNSEARCH;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(annsearch, loader) {
	duckdb::LoadInternal(loader);
}
}
