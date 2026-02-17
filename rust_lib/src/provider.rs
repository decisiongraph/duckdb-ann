//! In-memory DataProvider implementation for DiskANN.
//!
//! Uses flat contiguous vector storage for cache-friendly memory layout.
//! Adjacency lists are stored in a DashMap for concurrent insert safety.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use dashmap::DashMap;
use diskann::{
    ANNError, ANNResult,
    error::Infallible,
    graph::{AdjacencyList, glue},
    provider,
    utils::VectorRepr,
};
use diskann_vector::distance::Metric;
use parking_lot::RwLock;

// ==================
// Storage
// ==================

#[derive(Debug)]
struct Inner {
    /// Flat contiguous vector storage: [id*dim .. (id+1)*dim]
    vectors: RwLock<Vec<f32>>,
    /// Per-node adjacency lists (concurrent-safe for graph build)
    adjacency: DashMap<u32, AdjacencyList<u32>>,
    count: AtomicU32,
    start_point_ids: RwLock<Vec<u32>>,
    #[allow(dead_code)]
    max_degree: usize,
    dimension: usize,
    metric: Metric,
}

/// Newtype wrapper for the in-memory provider, allowing trait impls.
#[derive(Debug, Clone)]
pub struct Provider(Arc<Inner>);

impl Provider {
    pub fn new(dimension: usize, max_degree: usize, metric: Metric) -> Self {
        Self(Arc::new(Inner {
            vectors: RwLock::new(Vec::new()),
            adjacency: DashMap::new(),
            count: AtomicU32::new(0),
            start_point_ids: RwLock::new(Vec::new()),
            max_degree,
            dimension,
            metric,
        }))
    }

    /// Reconstruct a Provider from pre-existing data (for deserialization).
    pub fn bulk_load(
        dimension: usize,
        max_degree: usize,
        metric: Metric,
        flat_vectors: Vec<f32>,
        adjacency_lists: Vec<Vec<u32>>,
        entry_points: Vec<u32>,
        count: u32,
    ) -> Self {
        let inner = Arc::new(Inner {
            vectors: RwLock::new(flat_vectors),
            adjacency: DashMap::new(),
            count: AtomicU32::new(count),
            start_point_ids: RwLock::new(entry_points),
            max_degree,
            dimension,
            metric,
        });

        for (id, neighbors) in adjacency_lists.into_iter().enumerate() {
            let mut adj = AdjacencyList::new();
            adj.extend_from_slice(&neighbors);
            inner.adjacency.insert(id as u32, adj);
        }

        Self(inner)
    }

    pub fn len(&self) -> usize {
        self.0.count.load(Ordering::Relaxed) as usize
    }

    /// Insert as a start point. Called for the very first vector.
    pub fn insert_start_point(&self, id: u32, vector: Vec<f32>) {
        {
            let mut vecs = self.0.vectors.write();
            let offset = id as usize * self.0.dimension;
            if vecs.len() < offset + self.0.dimension {
                vecs.resize(offset + self.0.dimension, 0.0);
            }
            vecs[offset..offset + self.0.dimension].copy_from_slice(&vector);
        }
        self.0.adjacency.insert(id, AdjacencyList::new());
        self.0.count.fetch_max(id + 1, Ordering::Relaxed);
        self.0.start_point_ids.write().push(id);
    }

    /// Get a copy of the vector data for the given id.
    pub fn get_vector(&self, id: u32) -> Option<Vec<f32>> {
        let vecs = self.0.vectors.read();
        let offset = id as usize * self.0.dimension;
        if offset + self.0.dimension <= vecs.len() {
            Some(vecs[offset..offset + self.0.dimension].to_vec())
        } else {
            None
        }
    }

    /// Get a copy of the neighbor list for the given id.
    pub fn get_neighbors(&self, id: u32) -> Option<Vec<u32>> {
        self.0.adjacency.get(&id).map(|adj| adj.to_vec())
    }

    pub fn dim(&self) -> usize {
        self.0.dimension
    }

    pub fn metric(&self) -> Metric {
        self.0.metric
    }

    pub fn max_degree(&self) -> usize {
        self.0.max_degree
    }

    /// Write flat vectors to a writer (for serialization).
    pub fn write_vectors_to(&self, w: &mut dyn Write) -> std::io::Result<()> {
        let vecs = self.0.vectors.read();
        let count = self.0.count.load(Ordering::Relaxed) as usize;
        let total = count * self.0.dimension;
        let data = &vecs[..total];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, total * 4) };
        w.write_all(bytes)
    }

    /// Write fixed-width padded adjacency to a writer.
    /// Each node gets exactly `max_degree` u32 slots, unused padded with u32::MAX.
    pub fn write_adjacency_to(&self, w: &mut dyn Write, max_degree: usize) -> std::io::Result<()> {
        let count = self.0.count.load(Ordering::Relaxed) as usize;
        let sentinel = u32::MAX;
        let mut row = vec![sentinel; max_degree];
        for id in 0..count as u32 {
            // Fill row with sentinel
            row.fill(sentinel);
            if let Some(adj) = self.0.adjacency.get(&id) {
                let neighbors: &[u32] = &*adj;
                let n = neighbors.len().min(max_degree);
                row[..n].copy_from_slice(&neighbors[..n]);
            }
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(row.as_ptr() as *const u8, max_degree * 4)
            };
            w.write_all(bytes)?;
        }
        Ok(())
    }

    /// Expose start point IDs for serialization.
    pub fn get_entry_points(&self) -> Vec<u32> {
        self.0.start_point_ids.read().clone()
    }
}

// ==================
// Context
// ==================

#[derive(Debug, Clone)]
pub struct DefaultContext;

impl provider::ExecutionContext for DefaultContext {
    fn wrap_spawn<F, T>(&self, f: F) -> impl std::future::Future<Output = T> + Send + 'static
    where
        F: std::future::Future<Output = T> + Send + 'static,
    {
        f
    }
}

// ==================
// Error type
// ==================

#[derive(Debug, Clone, Copy)]
pub struct ProviderError(pub u32);

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid id {}", self.0)
    }
}

impl std::error::Error for ProviderError {}

impl From<ProviderError> for ANNError {
    #[track_caller]
    fn from(err: ProviderError) -> Self {
        ANNError::opaque(err)
    }
}

diskann::always_escalate!(ProviderError);

// ==================
// DataProvider
// ==================

impl provider::DataProvider for Provider {
    type Context = DefaultContext;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = ProviderError;

    fn to_internal_id(&self, _context: &DefaultContext, gid: &u32) -> Result<u32, ProviderError> {
        Ok(*gid)
    }

    fn to_external_id(&self, _context: &DefaultContext, id: u32) -> Result<u32, ProviderError> {
        Ok(id)
    }
}

// ==================
// Delete
// ==================

impl provider::Delete for Provider {
    async fn delete(
        &self,
        _context: &Self::Context,
        _gid: &Self::ExternalId,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn release(
        &self,
        _context: &Self::Context,
        _id: Self::InternalId,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn status_by_internal_id(
        &self,
        _context: &DefaultContext,
        id: u32,
    ) -> Result<provider::ElementStatus, Self::Error> {
        if (id as usize) < self.len() {
            Ok(provider::ElementStatus::Valid)
        } else {
            Err(ProviderError(id))
        }
    }

    fn status_by_external_id(
        &self,
        context: &DefaultContext,
        gid: &u32,
    ) -> impl std::future::Future<Output = Result<provider::ElementStatus, Self::Error>> + Send
    {
        self.status_by_internal_id(context, *gid)
    }
}

// ==================
// SetElement
// ==================

impl provider::SetElement<[f32]> for Provider {
    type SetError = ANNError;
    type Guard = provider::NoopGuard<u32>;

    async fn set_element(
        &self,
        _context: &DefaultContext,
        id: &u32,
        element: &[f32],
    ) -> Result<Self::Guard, Self::SetError> {
        {
            let mut vecs = self.0.vectors.write();
            let offset = *id as usize * self.0.dimension;
            if vecs.len() < offset + self.0.dimension {
                vecs.resize(offset + self.0.dimension, 0.0);
            }
            vecs[offset..offset + self.0.dimension].copy_from_slice(element);
        }
        self.0.adjacency.insert(*id, AdjacencyList::new());
        self.0.count.fetch_max(*id + 1, Ordering::Relaxed);
        Ok(provider::NoopGuard::new(*id))
    }
}

// ==================
// DefaultAccessor
// ==================

impl provider::DefaultAccessor for Provider {
    type Accessor<'a> = NeighborHandle<'a>;

    fn default_accessor(&self) -> Self::Accessor<'_> {
        NeighborHandle { inner: &self.0 }
    }
}

// ==================
// NeighborHandle (for NeighborAccessor / NeighborAccessorMut)
// ==================

#[derive(Debug, Clone, Copy)]
pub struct NeighborHandle<'a> {
    inner: &'a Inner,
}

impl provider::HasId for NeighborHandle<'_> {
    type Id = u32;
}

impl provider::NeighborAccessor for NeighborHandle<'_> {
    async fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> ANNResult<Self> {
        match self.inner.adjacency.get(&id) {
            Some(adj) => {
                neighbors.overwrite_trusted(&adj);
                Ok(self)
            }
            None => Err(ANNError::opaque(ProviderError(id))),
        }
    }
}

impl provider::NeighborAccessorMut for NeighborHandle<'_> {
    async fn set_neighbors(self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        match self.inner.adjacency.get_mut(&id) {
            Some(mut adj) => {
                adj.clear();
                adj.extend_from_slice(neighbors);
                Ok(self)
            }
            None => Err(ANNError::opaque(ProviderError(id))),
        }
    }

    async fn append_vector(self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        match self.inner.adjacency.get_mut(&id) {
            Some(mut adj) => {
                adj.extend_from_slice(neighbors);
                Ok(self)
            }
            None => Err(ANNError::opaque(ProviderError(id))),
        }
    }
}

// ==================
// Accessor (data accessor with buffer)
// ==================

#[derive(Debug)]
pub struct ProviderAccessor<'a> {
    inner: &'a Inner,
    buffer: Box<[f32]>,
}

impl<'a> ProviderAccessor<'a> {
    fn new(inner: &'a Inner) -> Self {
        let buffer = vec![0.0f32; inner.dimension].into_boxed_slice();
        Self { inner, buffer }
    }
}

impl provider::HasId for ProviderAccessor<'_> {
    type Id = u32;
}

impl provider::Accessor for ProviderAccessor<'_> {
    type Extended = Box<[f32]>;
    type Element<'e>
        = &'e [f32]
    where
        Self: 'e;
    type ElementRef<'e> = &'e [f32];
    type GetError = ProviderError;

    async fn get_element(&mut self, id: u32) -> Result<&[f32], ProviderError> {
        let vecs = self.inner.vectors.read();
        let offset = id as usize * self.inner.dimension;
        if offset + self.inner.dimension <= vecs.len() {
            self.buffer.copy_from_slice(&vecs[offset..offset + self.inner.dimension]);
            Ok(&*self.buffer)
        } else {
            Err(ProviderError(id))
        }
    }
}

impl<'a> provider::DelegateNeighbor<'a> for ProviderAccessor<'_> {
    type Delegate = NeighborHandle<'a>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        NeighborHandle { inner: self.inner }
    }
}

impl provider::BuildQueryComputer<[f32]> for ProviderAccessor<'_> {
    type QueryComputerError = Infallible;
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;

    fn build_query_computer(
        &self,
        from: &[f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(f32::query_distance(from, self.inner.metric))
    }
}

impl provider::BuildDistanceComputer for ProviderAccessor<'_> {
    type DistanceComputerError = Infallible;
    type DistanceComputer = <f32 as VectorRepr>::Distance;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(f32::distance(self.inner.metric, Some(self.inner.dimension)))
    }
}

// ==================
// Glue traits
// ==================

impl glue::SearchExt for ProviderAccessor<'_> {
    fn starting_points(&self) -> impl std::future::Future<Output = ANNResult<Vec<u32>>> + Send {
        let ids = self.inner.start_point_ids.read().clone();
        futures_util::future::ok(ids)
    }
}

impl glue::ExpandBeam<[f32]> for ProviderAccessor<'_> {}
impl glue::FillSet for ProviderAccessor<'_> {}

impl<'a> glue::AsElement<&'a [f32]> for ProviderAccessor<'a> {
    type Error = Infallible;
    fn as_element(
        &mut self,
        vector: &'a [f32],
        _id: Self::Id,
    ) -> impl std::future::Future<Output = Result<Self::Element<'_>, Self::Error>> + Send {
        std::future::ready(Ok(vector))
    }
}

// ==================
// Strategy
// ==================

#[derive(Debug, Default, Clone, Copy)]
pub struct FullPrecisionStrategy;

impl FullPrecisionStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl glue::SearchStrategy<Provider, [f32]> for FullPrecisionStrategy {
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;
    type PostProcessor = glue::CopyIds;
    type SearchAccessorError = Infallible;
    type SearchAccessor<'a> = ProviderAccessor<'a>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        _context: &'a DefaultContext,
    ) -> Result<ProviderAccessor<'a>, Infallible> {
        Ok(ProviderAccessor::new(&provider.0))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl glue::PruneStrategy<Provider> for FullPrecisionStrategy {
    type DistanceComputer = <f32 as VectorRepr>::Distance;
    type PruneAccessor<'a> = ProviderAccessor<'a>;
    type PruneAccessorError = Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        _context: &'a DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(ProviderAccessor::new(&provider.0))
    }
}

impl glue::InsertStrategy<Provider, [f32]> for FullPrecisionStrategy {
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        _context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(ProviderAccessor::new(&provider.0))
    }
}
