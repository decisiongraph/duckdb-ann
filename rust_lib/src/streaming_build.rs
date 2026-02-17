//! Two-pass streaming DiskANN builder for datasets larger than RAM.
//!
//! Input: flat binary file `[u32 num_vectors][u32 dimension][f32*N*D]`
//! Output: .diskann index file
//!
//! Pass 1 (sample): Read a subset of vectors, build an in-memory pilot graph.
//! Pass 2 (stream): For each remaining vector, greedy-search the pilot graph
//!   to find approximate neighbors. Write all vectors + adjacency to disk.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use anyhow::{anyhow, Result};

use crate::file_format::{MAGIC, VERSION};
use crate::index_manager::Metric;

/// Header for the input vectors binary file.
struct VecFileHeader {
    num_vectors: u32,
    dimension: u32,
}

fn read_vec_header(r: &mut impl Read) -> io::Result<VecFileHeader> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    let num_vectors = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let dimension = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    Ok(VecFileHeader { num_vectors, dimension })
}

/// Read a single vector (dimension floats) from the reader.
fn read_vector(r: &mut impl Read, dim: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; dim * 4];
    r.read_exact(&mut bytes)?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(floats)
}

/// Build a DiskANN index from a binary vectors file using streaming two-pass approach.
///
/// Only the sample vectors + their graph stay in RAM. Remaining vectors are
/// processed one at a time from disk.
pub fn streaming_build(
    input_path: &str,
    output_path: &str,
    metric: Metric,
    max_degree: u32,
    build_complexity: u32,
    alpha: f32,
    sample_size: u32,
) -> Result<StreamingBuildResult> {
    let input = File::open(input_path)
        .map_err(|e| anyhow!("Failed to open input '{}': {}", input_path, e))?;
    let mut reader = BufReader::new(input);

    let hdr = read_vec_header(&mut reader)
        .map_err(|e| anyhow!("Failed to read input header: {}", e))?;

    if hdr.num_vectors == 0 {
        return Err(anyhow!("Input file has 0 vectors"));
    }
    if hdr.dimension == 0 {
        return Err(anyhow!("Input file has dimension 0"));
    }

    let dim = hdr.dimension as usize;
    let n = hdr.num_vectors;
    // Auto sample size: max(sqrt(N), 1000), clamped to N
    let sample_n = if sample_size == 0 {
        ((n as f64).sqrt() as usize).max(1000).min(n as usize)
    } else {
        (sample_size as usize).min(n as usize)
    };
    let deg = max_degree as usize;

    // ========================================
    // Pass 1: Build pilot graph from sample
    // ========================================

    // Create in-memory index for the sample
    let pilot = crate::index_manager::InMemoryIndex::new_detached(
        dim,
        metric,
        max_degree,
        build_complexity,
        alpha,
    );

    // Read and insert sample vectors (first sample_n vectors)
    let mut sample_vectors: Vec<Vec<f32>> = Vec::with_capacity(sample_n);
    for _i in 0..sample_n {
        let vec = read_vector(&mut reader, dim)
            .map_err(|e| anyhow!("Failed to read sample vector: {}", e))?;
        sample_vectors.push(vec);
    }

    // Insert all sample vectors into the pilot graph
    for vec in &sample_vectors {
        pilot.add(vec)?;
    }

    // Get adjacency lists for sample vectors from the pilot graph (mutable for back-edge injection)
    let mut sample_adj = pilot.get_all_adjacency(sample_n, deg);

    // ========================================
    // Pass 2: Stream remaining vectors, find approximate neighbors
    // ========================================

    // For non-sample vectors, we search the pilot graph AND a growing secondary
    // index of previously inserted streaming vectors. This builds both
    // streaming→sample and streaming→streaming edges, improving recall.

    let mut stream_adj: Vec<Vec<u32>> = Vec::with_capacity((n as usize).saturating_sub(sample_n));
    let mut stream_vectors: Vec<Vec<f32>> = Vec::with_capacity((n as usize).saturating_sub(sample_n));

    // Secondary index for streaming vectors (enables streaming→streaming edges)
    let stream_index = crate::index_manager::InMemoryIndex::new_detached(
        dim,
        metric,
        max_degree,
        build_complexity,
        alpha,
    );

    let remaining = n as usize - sample_n;
    for i in 0..remaining {
        let vec = read_vector(&mut reader, dim)
            .map_err(|e| anyhow!("Failed to read streaming vector: {}", e))?;

        // Search pilot graph for nearest sample vectors
        let pilot_results = pilot.search(&vec, deg, build_complexity)?;

        // Search secondary index for nearest streaming vectors (if non-empty)
        let stream_results = if i > 0 {
            stream_index.search(&vec, deg, build_complexity).unwrap_or_default()
        } else {
            Vec::new()
        };

        // Merge results: remap streaming index labels to global IDs, then
        // take the top-k by distance from the combined set
        let mut combined: Vec<(u32, f32)> = Vec::with_capacity(pilot_results.len() + stream_results.len());
        for (id, dist) in &pilot_results {
            combined.push((*id as u32, *dist));
        }
        for (label, dist) in &stream_results {
            // Streaming index labels are 0-based sequential; global ID = sample_n + label
            let global_id = sample_n as u32 + *label as u32;
            combined.push((global_id, *dist));
        }
        combined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        combined.dedup_by_key(|x| x.0);
        combined.truncate(deg);

        let neighbors: Vec<u32> = combined.iter().map(|(id, _)| *id).collect();
        stream_adj.push(neighbors);

        // Add this vector to the secondary index for future streaming vectors
        stream_vectors.push(vec.clone());
        let _ = stream_index.add(&vec);
    }

    // ========================================
    // Back-edge injection: make streaming vectors reachable
    // ========================================
    // Without back-edges, streaming vectors only have forward edges TO neighbors
    // but those neighbors may not point back TO them, making them unreachable.
    // Fix: for each streaming vector, inject its global ID into one of its
    // neighbors' adjacency lists (both sample and streaming neighbors).

    // Collect back-edge targets first to avoid borrow conflicts
    let back_edges: Vec<(usize, u32, u32)> = (0..stream_adj.len())
        .filter_map(|i| {
            let adj = &stream_adj[i];
            if adj.is_empty() {
                return None;
            }
            let stream_global_id = (sample_n + i) as u32;
            let target_id = adj[i % adj.len()];
            Some((i, stream_global_id, target_id))
        })
        .collect();

    for (i, stream_global_id, target_id) in back_edges {
        if (target_id as usize) < sample_n {
            // Target is a sample vector
            let sample_neighbors = &mut sample_adj[target_id as usize];
            if sample_neighbors.len() < deg {
                sample_neighbors.push(stream_global_id);
            } else {
                let pos = stream_global_id as usize % deg;
                sample_neighbors[pos] = stream_global_id;
            }
        } else {
            // Target is a streaming vector — inject back-edge into its adjacency
            let stream_idx = target_id as usize - sample_n;
            if stream_idx < stream_adj.len() && stream_idx != i {
                if !stream_adj[stream_idx].contains(&stream_global_id) {
                    if stream_adj[stream_idx].len() < deg {
                        stream_adj[stream_idx].push(stream_global_id);
                    } else {
                        let pos = stream_global_id as usize % deg;
                        stream_adj[stream_idx][pos] = stream_global_id;
                    }
                }
            }
        }
    }

    // ========================================
    // Write output .diskann file
    // ========================================

    // Entry points: use the pilot graph's entry points (they're sample vector IDs)
    let entry_points = pilot.get_entry_points();
    let num_entry_points = entry_points.len() as u32;

    let output = File::create(output_path)
        .map_err(|e| anyhow!("Failed to create output '{}': {}", output_path, e))?;
    let mut writer = BufWriter::new(output);

    let metric_byte = match metric {
        Metric::L2 => 0u8,
        Metric::InnerProduct => 1u8,
    };

    // Write header (32 bytes)
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&n.to_le_bytes())?;
    writer.write_all(&(dim as u32).to_le_bytes())?;
    writer.write_all(&max_degree.to_le_bytes())?;
    writer.write_all(&num_entry_points.to_le_bytes())?;
    writer.write_all(&[metric_byte])?;
    writer.write_all(&[0u8; 3])?; // padding
    writer.write_all(&build_complexity.to_le_bytes())?;

    // Write entry point IDs
    for &ep in &entry_points {
        writer.write_all(&ep.to_le_bytes())?;
    }

    // Write all vectors: re-read from input file
    // Seek back to start of vectors in input
    let input2 = File::open(input_path)?;
    let mut reader2 = BufReader::new(input2);
    reader2.seek(SeekFrom::Start(8))?; // skip header

    // Copy all vectors directly
    let total_vec_bytes = n as usize * dim * 4;
    let mut remaining_bytes = total_vec_bytes;
    let mut buf = vec![0u8; 64 * 1024]; // 64KB copy buffer
    while remaining_bytes > 0 {
        let to_read = remaining_bytes.min(buf.len());
        reader2.read_exact(&mut buf[..to_read])?;
        writer.write_all(&buf[..to_read])?;
        remaining_bytes -= to_read;
    }

    // Write adjacency lists
    let sentinel = u32::MAX;
    let mut row = vec![sentinel; deg];

    // Sample vectors: use pilot graph adjacency
    for i in 0..sample_n {
        row.fill(sentinel);
        let adj = &sample_adj[i];
        let copy_n = adj.len().min(deg);
        row[..copy_n].copy_from_slice(&adj[..copy_n]);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(row.as_ptr() as *const u8, deg * 4)
        };
        writer.write_all(bytes)?;
    }

    // Streaming vectors: use approximate neighbors from pilot search
    for adj in &stream_adj {
        row.fill(sentinel);
        let copy_n = adj.len().min(deg);
        row[..copy_n].copy_from_slice(&adj[..copy_n]);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(row.as_ptr() as *const u8, deg * 4)
        };
        writer.write_all(bytes)?;
    }

    writer.flush()?;

    Ok(StreamingBuildResult {
        num_vectors: n,
        dimension: dim as u32,
        sample_size: sample_n as u32,
    })
}

pub struct StreamingBuildResult {
    pub num_vectors: u32,
    pub dimension: u32,
    pub sample_size: u32,
}
