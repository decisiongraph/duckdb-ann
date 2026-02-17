# annsearch — ANN Vector Indexes for DuckDB

DuckDB extension providing approximate nearest neighbor (ANN) vector indexes using **DiskANN** and **FAISS**. Indexes are stored inside the `.duckdb` file, survive restarts, and integrate with the query optimizer.

## Quick Start

```sql
-- Create a table with vector embeddings
CREATE TABLE docs (id INT, embedding FLOAT[384]);
INSERT INTO docs SELECT i, [random()::FLOAT for _ in range(384)] FROM range(10000) t(i);

-- Create an ANN index
CREATE INDEX docs_ann ON docs USING DISKANN (embedding);

-- Search automatically uses the index
SELECT id FROM docs ORDER BY array_distance(embedding, [0.1, 0.2, ...]::FLOAT[384]) LIMIT 10;
```

## Index Types

### DISKANN

Graph-based ANN index using the [DiskANN](https://github.com/microsoft/DiskANN) algorithm (Rust implementation). Good general-purpose choice.

```sql
CREATE INDEX idx ON table USING DISKANN (column) WITH (
    metric       = 'L2',    -- distance metric (L2 or IP)
    max_degree   = 64,      -- graph connectivity
    build_complexity = 128,  -- build-time search width (higher = better quality)
    alpha        = 1.2,      -- pruning expansion factor
    quantization = 'sq8'     -- optional: 8-bit scalar quantization (~4x memory reduction)
);
```

### FAISS

Wraps [FAISS](https://github.com/facebookresearch/faiss) indexes. Supports multiple index structures and optional GPU acceleration.

```sql
-- Flat (exact, no approximation)
CREATE INDEX idx ON table USING FAISS (column) WITH (type='Flat');

-- HNSW (graph-based, fast search)
CREATE INDEX idx ON table USING FAISS (column) WITH (type='HNSW', hnsw_m=32);

-- IVFFlat (inverted file, good for large datasets)
CREATE INDEX idx ON table USING FAISS (column) WITH (
    type='IVFFlat', ivf_nlist=100, nprobe=4, train_sample=50000
);

-- GPU-accelerated (Metal on macOS)
CREATE INDEX idx ON table USING FAISS (column) WITH (type='Flat', gpu=true);
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `metric` | VARCHAR | `'L2'` | `'L2'` or `'IP'` (inner product) |
| `type` | VARCHAR | `'Flat'` | `'Flat'`, `'HNSW'`, or `'IVFFlat'` |
| `hnsw_m` | INTEGER | 32 | HNSW graph connectivity |
| `ivf_nlist` | INTEGER | 100 | Number of IVF centroids |
| `nprobe` | INTEGER | 1 | IVF partitions to probe at search time |
| `train_sample` | INTEGER | 0 | Vectors for IVF training (0 = all) |
| `description` | VARCHAR | | FAISS `index_factory` string (advanced, overrides `type`) |
| `gpu` | BOOLEAN | false | Upload index to GPU for search |

## GPU Acceleration

FAISS indexes support GPU-accelerated search on macOS via Metal. The GPU backend handles both `IndexFlat` and `IndexIVFFlat`.

```sql
-- Check GPU availability
SELECT * FROM faiss_gpu_info();
-- available | device
-- true      | Metal GPU (family=9)

-- Create a GPU-accelerated index
CREATE INDEX gpu_idx ON docs USING FAISS (embedding) WITH (gpu=true);
```

**How it works:**
- A CPU copy of the index is always maintained (for inserts and serialization)
- A GPU copy is lazily created on first search
- Inserts invalidate the GPU copy; it's rebuilt on the next search
- The `gpu` flag is persisted, so the index re-uploads on database reopen
- Falls back to CPU transparently if no GPU is available

**Requirements:** Build with `FAISS_AVAILABLE` and `FAISS_METAL_ENABLED` (requires Xcode + faiss-metal).

## Automatic Optimizer

Queries matching `ORDER BY distance_fn(col, query) LIMIT k` automatically use the index:

```sql
-- All of these use the index automatically:
SELECT * FROM docs ORDER BY array_distance(embedding, ?::FLOAT[384]) LIMIT 10;
SELECT * FROM docs ORDER BY array_inner_product(embedding, ?::FLOAT[384]) LIMIT 10;
SELECT * FROM docs ORDER BY array_cosine_similarity(embedding, ?::FLOAT[384]) LIMIT 10;
```

Supported distance functions: `array_distance`, `list_distance`, `array_inner_product`, `list_inner_product`, `array_cosine_similarity`, `list_cosine_similarity`.

When a `WHERE` clause is present, the optimizer overfetches (`3x + 100`) to compensate for post-filtering.

## Table Functions

### `ann_search` — Search with row fetch

```sql
SELECT * FROM ann_search('docs', 'docs_ann', [0.1, 0.2, ...]::FLOAT[384], 10);
-- Returns: all table columns + _distance, ordered by distance

-- Named parameters:
SELECT * FROM ann_search('docs', 'docs_ann', query, 10, search_complexity := 256, oversample := 3);
```

### `ann_search_batch` — Multi-query batch search

```sql
SELECT * FROM ann_search_batch('docs', 'docs_ann',
    [[0.1, 0.2, ...], [0.3, 0.4, ...]]::LIST(FLOAT[384]), 5);
-- Returns: query_idx, table columns, _distance
```

### `diskann_index_scan` / `faiss_index_scan` — Low-level index scan

```sql
SELECT row_id, distance FROM diskann_index_scan('docs', 'docs_ann', [0.1, ...]::FLOAT[384], 10);
-- Returns: (BIGINT row_id, FLOAT distance)
```

### `annsearch_list` / `annsearch_index_info` — Diagnostics

```sql
SELECT * FROM annsearch_list();
-- name | engine  | table_name

SELECT * FROM annsearch_index_info();
-- name | engine | table_name | num_vectors | num_deleted | memory_bytes | quantized
```

### `diskann_streaming_build` — Build from binary file

Two-pass streaming build for datasets larger than RAM:

```sql
SELECT * FROM diskann_streaming_build('/tmp/vectors.bin', '/tmp/index.diskann',
    metric := 'l2', max_degree := 64, sample_size := 10000);
```

Input format: `[u32 num_vectors][u32 dimension][f32 * N * D]` (little-endian).

## Building

```bash
# Clone with submodules
git clone --recursive https://github.com/decisiongraph/duckdb-annsearch
cd duckdb-annsearch

# Build (DiskANN always, FAISS if available)
make release GEN=ninja

# Run tests
make test
```

**Requirements:** C++17 compiler, CMake, Rust toolchain. Optional: FAISS (`-DENABLE_FAISS=ON`), Metal GPU support.

If using [devenv](https://devenv.sh/), `devenv shell` provides all dependencies.
