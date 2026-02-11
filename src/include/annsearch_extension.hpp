#pragma once

#include "duckdb.hpp"

namespace duckdb {

class AnnsearchExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

// DiskANN function registration (always available)
void RegisterDiskannCreateFunction(ExtensionLoader &loader);
void RegisterDiskannDestroyFunction(ExtensionLoader &loader);
void RegisterDiskannAddFunction(ExtensionLoader &loader);
void RegisterDiskannSearchFunction(ExtensionLoader &loader);
void RegisterDiskannListFunction(ExtensionLoader &loader);
void RegisterDiskannInfoFunction(ExtensionLoader &loader);

// Unified listing
void RegisterAnnsearchListFunction(ExtensionLoader &loader);

#ifdef FAISS_AVAILABLE
// FAISS function registration (conditional on libfaiss)
void RegisterFaissCreateFunction(ExtensionLoader &loader);
void RegisterFaissAddFunction(ExtensionLoader &loader);
void RegisterFaissSearchFunction(ExtensionLoader &loader);
void RegisterFaissPersistFunctions(ExtensionLoader &loader);
void RegisterFaissManageFunctions(ExtensionLoader &loader);
void RegisterFaissGpuFunctions(ExtensionLoader &loader);
#endif

} // namespace duckdb
