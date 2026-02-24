-- HNSW Apple Silicon benchmark
-- Tests FAISS HNSW create + search with Accelerate/Metal optimizations
-- Run from build/release: ./duckdb -f ../../bench/hnsw_apple_silicon.sql

.timer on

-- ========================================
-- Setup: 10K vectors, 768-dim
-- ========================================

CREATE TABLE bench_10k (id INT, embedding FLOAT[768]);

INSERT INTO bench_10k
SELECT i, list_transform(generate_series(1, 768), x -> (random() - 0.5)::FLOAT)::FLOAT[768]
FROM generate_series(1, 10000) t(i);

-- Use first vector as query
SET VARIABLE q_768 = (SELECT embedding FROM bench_10k LIMIT 1);

SELECT '--- 10K vectors, 768-dim ---' AS benchmark;

SELECT 'HNSW create (10K)' AS step;
CREATE INDEX bench_10k_hnsw ON bench_10k USING FAISS (embedding) WITH (type='HNSW', hnsw_m=32);

SELECT 'HNSW search k=10 (10K)' AS step;
SELECT row_id, distance FROM faiss_index_scan('bench_10k', 'bench_10k_hnsw', getvariable('q_768'), 10);

SELECT 'HNSW search k=100 (10K)' AS step;
SELECT count(*) FROM faiss_index_scan('bench_10k', 'bench_10k_hnsw', getvariable('q_768'), 100);

DROP INDEX bench_10k_hnsw;
DROP TABLE bench_10k;

-- ========================================
-- Setup: 50K vectors, 768-dim
-- ========================================

CREATE TABLE bench_50k (id INT, embedding FLOAT[768]);

INSERT INTO bench_50k
SELECT i, list_transform(generate_series(1, 768), x -> (random() - 0.5)::FLOAT)::FLOAT[768]
FROM generate_series(1, 50000) t(i);

SET VARIABLE q_768 = (SELECT embedding FROM bench_50k LIMIT 1);

SELECT '--- 50K vectors, 768-dim ---' AS benchmark;

SELECT 'HNSW create (50K)' AS step;
CREATE INDEX bench_50k_hnsw ON bench_50k USING FAISS (embedding) WITH (type='HNSW', hnsw_m=32);

SELECT 'HNSW search k=10 (50K)' AS step;
SELECT row_id, distance FROM faiss_index_scan('bench_50k', 'bench_50k_hnsw', getvariable('q_768'), 10);

SELECT 'HNSW search k=100 (50K)' AS step;
SELECT count(*) FROM faiss_index_scan('bench_50k', 'bench_50k_hnsw', getvariable('q_768'), 100);

DROP INDEX bench_50k_hnsw;
DROP TABLE bench_50k;

-- ========================================
-- Setup: 100K vectors, 768-dim
-- ========================================

CREATE TABLE bench_100k (id INT, embedding FLOAT[768]);

INSERT INTO bench_100k
SELECT i, list_transform(generate_series(1, 768), x -> (random() - 0.5)::FLOAT)::FLOAT[768]
FROM generate_series(1, 100000) t(i);

SET VARIABLE q_768 = (SELECT embedding FROM bench_100k LIMIT 1);

SELECT '--- 100K vectors, 768-dim ---' AS benchmark;

SELECT 'HNSW create (100K)' AS step;
CREATE INDEX bench_100k_hnsw ON bench_100k USING FAISS (embedding) WITH (type='HNSW', hnsw_m=32);

SELECT 'HNSW search k=10 (100K)' AS step;
SELECT row_id, distance FROM faiss_index_scan('bench_100k', 'bench_100k_hnsw', getvariable('q_768'), 10);

SELECT 'HNSW search k=100 (100K)' AS step;
SELECT count(*) FROM faiss_index_scan('bench_100k', 'bench_100k_hnsw', getvariable('q_768'), 100);

-- ========================================
-- DiskANN comparison (same 100K data)
-- ========================================

SELECT '--- DiskANN comparison (100K) ---' AS benchmark;

SELECT 'DiskANN create (100K)' AS step;
CREATE INDEX bench_100k_diskann ON bench_100k USING DISKANN (embedding);

SELECT 'DiskANN search k=10 (100K)' AS step;
SELECT row_id, distance FROM diskann_index_scan('bench_100k', 'bench_100k_diskann', getvariable('q_768'), 10);

SELECT 'DiskANN search k=100 (100K)' AS step;
SELECT count(*) FROM diskann_index_scan('bench_100k', 'bench_100k_diskann', getvariable('q_768'), 100);

-- Cleanup
DROP INDEX bench_100k_hnsw;
DROP INDEX bench_100k_diskann;
DROP TABLE bench_100k;

SELECT 'Benchmark complete' AS status;
