use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::{Arc, LazyLock};

/// Global index registry.
static INDEXES: LazyLock<DashMap<String, Arc<ManagedIndex>>> = LazyLock::new(DashMap::new);

/// Tokio runtime for async diskann operations.
static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

pub fn runtime() -> &'static tokio::runtime::Runtime {
    &RUNTIME
}

/// In-memory vector store wrapping DiskANN.
///
/// NOTE: The `diskann` crate (v0.45) has a heavily generic DataProvider trait system.
/// For the initial implementation, we use a simple Vec-based store and build
/// DiskANN's graph on top. The DataProvider integration can be refined as the
/// crate's API stabilizes.
pub struct ManagedIndex {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub max_degree: u32,
    pub build_complexity: u32,
    pub alpha: f32,
    // Raw vector storage: vectors[i] has dimension floats starting at i * dimension.
    vectors: RwLock<Vec<f32>>,
    // Whether the graph has been built.
    built: RwLock<bool>,
    // TODO: Replace brute-force with actual DiskANN graph index once DataProvider is implemented.
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    L2,
    InnerProduct,
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Metric::L2 => write!(f, "L2"),
            Metric::InnerProduct => write!(f, "IP"),
        }
    }
}

impl ManagedIndex {
    fn new(
        name: String,
        dimension: usize,
        metric: Metric,
        max_degree: u32,
        build_complexity: u32,
        alpha: f32,
    ) -> Self {
        Self {
            name,
            dimension,
            metric,
            max_degree,
            build_complexity,
            alpha,
            vectors: RwLock::new(Vec::new()),
            built: RwLock::new(false),
        }
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.vectors.read().len() / self.dimension
    }

    /// Add a single vector. Returns the assigned label.
    pub fn add(&self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Expected dimension {}, got {}",
                self.dimension,
                vector.len()
            ));
        }
        let mut vecs = self.vectors.write();
        let label = vecs.len() / self.dimension;
        vecs.extend_from_slice(vector);
        // Invalidate build
        *self.built.write() = false;
        Ok(label as u64)
    }

    /// Brute-force k-NN search.
    ///
    /// TODO: Once DiskANN graph is integrated, use graph-based search.
    /// For now, brute-force ensures correctness and allows testing the SQL interface.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                self.dimension
            ));
        }

        let vecs = self.vectors.read();
        let n = vecs.len() / self.dimension;
        if n == 0 {
            return Ok(Vec::new());
        }

        let k = k.min(n);

        // Compute distances
        let mut dists: Vec<(u64, f32)> = (0..n)
            .map(|i| {
                let start = i * self.dimension;
                let v = &vecs[start..start + self.dimension];
                let dist = match self.metric {
                    Metric::L2 => l2_distance(query, v),
                    Metric::InnerProduct => -inner_product(query, v), // negate for min-heap
                };
                (i as u64, dist)
            })
            .collect();

        // Partial sort for top-k
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);

        // For IP metric, negate back
        if self.metric == Metric::InnerProduct {
            for d in &mut dists {
                d.1 = -d.1;
            }
        }

        Ok(dists)
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ========================================
// Public API for managing indexes
// ========================================

pub fn create_index(
    name: &str,
    dimension: usize,
    metric: Metric,
    max_degree: u32,
    build_complexity: u32,
    alpha: f32,
) -> Result<()> {
    if INDEXES.contains_key(name) {
        return Err(anyhow!("Index '{}' already exists", name));
    }
    let index = ManagedIndex::new(
        name.to_string(),
        dimension,
        metric,
        max_degree,
        build_complexity,
        alpha,
    );
    INDEXES.insert(name.to_string(), Arc::new(index));
    Ok(())
}

pub fn destroy_index(name: &str) -> Result<()> {
    INDEXES
        .remove(name)
        .ok_or_else(|| anyhow!("Index '{}' not found", name))?;
    Ok(())
}

pub fn get_index(name: &str) -> Result<Arc<ManagedIndex>> {
    INDEXES
        .get(name)
        .map(|r| r.value().clone())
        .ok_or_else(|| anyhow!("Index '{}' not found", name))
}

#[derive(Debug)]
pub struct IndexInfo {
    pub name: String,
    pub dimension: usize,
    pub count: usize,
    pub metric: Metric,
    pub max_degree: u32,
}

pub fn list_indexes() -> Vec<IndexInfo> {
    INDEXES
        .iter()
        .map(|entry| {
            let idx = entry.value();
            IndexInfo {
                name: idx.name.clone(),
                dimension: idx.dimension,
                count: idx.len(),
                metric: idx.metric,
                max_degree: idx.max_degree,
            }
        })
        .collect()
}

pub fn get_info(name: &str) -> Result<IndexInfo> {
    let idx = get_index(name)?;
    Ok(IndexInfo {
        name: idx.name.clone(),
        dimension: idx.dimension,
        count: idx.len(),
        metric: idx.metric,
        max_degree: idx.max_degree,
    })
}
