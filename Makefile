PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=annsearch
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Override: only run diskann + annsearch tests by default (always work).
# FAISS tests are in test/sql_faiss/ — run them with: make test_faiss
TESTS_BASE_DIRECTORY = "test/sql/"

# Run FAISS-specific tests (requires FAISS to be installed)
test_faiss:
	./build/release/$(TEST_PATH) "test/sql_faiss/*"
