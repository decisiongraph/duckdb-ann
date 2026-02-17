//! Binary .diskann file format for serialized DiskANN indexes.
//!
//! Layout (v2, 32-byte header):
//! [Header: 32 bytes]
//!   magic: "DANN" (4 bytes)
//!   version: u32 (=2)
//!   num_vectors: u32
//!   dimension: u32
//!   max_degree: u32
//!   num_entry_points: u32
//!   metric: u8 (0=L2, 1=IP)
//!   _pad: [u8; 3]
//!   build_complexity: u32
//! [Entry point IDs: num_entry_points * 4 bytes]
//! [Vector segment: num_vectors * dimension * 4 bytes]
//! [Adjacency segment: num_vectors * max_degree * 4 bytes]
//!   - Unused slots padded with u32::MAX sentinel
//!   - All values little-endian

use std::io::Write;

use crate::index_manager::Metric;
use crate::provider::Provider;

pub const MAGIC: &[u8; 4] = b"DANN";
pub const VERSION: u32 = 2;
pub const HEADER_SIZE: usize = 32;

#[derive(Debug, Clone)]
pub struct FileHeader {
    pub num_vectors: u32,
    pub dimension: u32,
    pub max_degree: u32,
    pub num_entry_points: u32,
    pub metric: u8,
    pub build_complexity: u32,
}

impl FileHeader {
    pub fn entry_points_offset(&self) -> usize {
        HEADER_SIZE
    }

    pub fn entry_points_size(&self) -> usize {
        self.num_entry_points as usize * 4
    }

    pub fn vectors_offset(&self) -> usize {
        self.entry_points_offset() + self.entry_points_size()
    }

    pub fn vectors_size(&self) -> usize {
        self.num_vectors as usize * self.dimension as usize * 4
    }

    pub fn adjacency_offset(&self) -> usize {
        self.vectors_offset() + self.vectors_size()
    }

    pub fn adjacency_size(&self) -> usize {
        self.num_vectors as usize * self.max_degree as usize * 4
    }

    pub fn total_file_size(&self) -> usize {
        self.adjacency_offset() + self.adjacency_size()
    }

    pub fn metric_enum(&self) -> Metric {
        match self.metric {
            1 => Metric::InnerProduct,
            _ => Metric::L2,
        }
    }
}

fn metric_to_u8(m: Metric) -> u8 {
    match m {
        Metric::L2 => 0,
        Metric::InnerProduct => 1,
    }
}

/// Write a complete .diskann index file.
pub fn write_index(
    w: &mut dyn Write,
    provider: &Provider,
    metric: Metric,
    build_complexity: u32,
) -> std::io::Result<()> {
    let entry_points = provider.get_entry_points();
    let num_vectors = provider.len() as u32;
    let dimension = provider.dim() as u32;
    let max_degree = provider.max_degree() as u32;
    let num_entry_points = entry_points.len() as u32;

    // Write header (32 bytes)
    w.write_all(MAGIC)?;                                // 4
    w.write_all(&VERSION.to_le_bytes())?;               // 4
    w.write_all(&num_vectors.to_le_bytes())?;           // 4
    w.write_all(&dimension.to_le_bytes())?;             // 4
    w.write_all(&max_degree.to_le_bytes())?;            // 4
    w.write_all(&num_entry_points.to_le_bytes())?;      // 4
    w.write_all(&[metric_to_u8(metric)])?;              // 1
    w.write_all(&[0u8; 3])?;                            // 3 pad
    w.write_all(&build_complexity.to_le_bytes())?;      // 4
    // total: 32

    // Write entry point IDs
    for id in &entry_points {
        w.write_all(&id.to_le_bytes())?;
    }

    // Write flat vectors
    provider.write_vectors_to(w)?;

    // Write adjacency
    provider.write_adjacency_to(w, max_degree as usize)?;

    Ok(())
}
