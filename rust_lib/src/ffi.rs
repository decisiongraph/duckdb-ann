//! C FFI interface for the DiskANN index manager.
//! Called from the C++ DuckDB extension.

use crate::index_manager::{self, InMemoryIndex, Metric};
use std::ffi::{c_char, CStr, CString};
use std::ptr;

// ========================================
// JSON-based result (kept for list/info)
// ========================================

/// FFI-safe result: either json_ptr or error_ptr is set.
/// Caller must free with `diskann_free_result`.
#[repr(C)]
pub struct DiskannResult {
    pub json_ptr: *mut c_char,
    pub error_ptr: *mut c_char,
}

fn ok_result(json: String) -> DiskannResult {
    DiskannResult {
        json_ptr: string_to_ptr(json),
        error_ptr: ptr::null_mut(),
    }
}

fn err_result(msg: String) -> DiskannResult {
    DiskannResult {
        json_ptr: ptr::null_mut(),
        error_ptr: string_to_ptr(msg),
    }
}

fn string_to_ptr(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a DiskannResult (both pointers).
#[no_mangle]
pub unsafe extern "C" fn diskann_free_result(result: DiskannResult) {
    if !result.json_ptr.is_null() {
        drop(CString::from_raw(result.json_ptr));
    }
    if !result.error_ptr.is_null() {
        drop(CString::from_raw(result.error_ptr));
    }
}

// ========================================
// Buffer-based helpers
// ========================================

/// Write a null-terminated error message into the caller's buffer.
/// Truncates if msg is longer than buf_len - 1.
unsafe fn write_err(buf: *mut c_char, buf_len: i32, msg: &str) {
    if buf.is_null() || buf_len <= 0 {
        return;
    }
    let max = (buf_len - 1) as usize;
    let bytes = msg.as_bytes();
    let copy_len = bytes.len().min(max);
    ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy_len);
    *buf.add(copy_len) = 0;
}

/// Helper to convert *const c_char to &str, writing error to buf on failure.
/// Returns None if conversion fails (error already written).
unsafe fn cstr_to_str<'a>(
    p: *const c_char,
    label: &str,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> Option<&'a str> {
    match CStr::from_ptr(p).to_str() {
        Ok(s) => Some(s),
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("Invalid {}: {}", label, e));
            None
        }
    }
}

// ========================================
// Buffer-based FFI functions (hot + cold paths)
// ========================================

/// Search: fills caller-provided output buffers.
/// Returns number of results written (0..=k), or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_search_buf(
    name: *const c_char,
    query_ptr: *const f32,
    dimension: i32,
    k: i32,
    search_complexity: i32,
    out_labels: *mut i64,
    out_distances: *mut f32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    let name = match cstr_to_str(name, "name", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };

    if dimension <= 0 {
        write_err(err_buf, err_buf_len, &format!("Invalid query dimension: {}", dimension));
        return -1;
    }
    if query_ptr.is_null() {
        write_err(err_buf, err_buf_len, "Null query pointer");
        return -1;
    }
    if out_labels.is_null() || out_distances.is_null() {
        write_err(err_buf, err_buf_len, "Null output buffer");
        return -1;
    }

    let idx = match index_manager::get_index(name) {
        Ok(idx) => idx,
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            return -1;
        }
    };

    if dimension as usize != idx.dimension() {
        write_err(
            err_buf,
            err_buf_len,
            &format!(
                "Dimension mismatch: query has {} but index expects {}",
                dimension,
                idx.dimension()
            ),
        );
        return -1;
    }

    let query = std::slice::from_raw_parts(query_ptr, dimension as usize);

    match idx.search(query, k as usize, search_complexity as u32) {
        Ok(results) => {
            let n = results.len().min(k as usize);
            for i in 0..n {
                *out_labels.add(i) = results[i].0 as i64;
                *out_distances.add(i) = results[i].1;
            }
            n as i32
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            -1
        }
    }
}

/// Add vector: returns assigned label, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_add_vector_buf(
    name: *const c_char,
    vector_ptr: *const f32,
    dimension: i32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i64 {
    let name = match cstr_to_str(name, "name", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };

    if dimension <= 0 {
        write_err(err_buf, err_buf_len, &format!("Invalid dimension: {}", dimension));
        return -1;
    }
    if vector_ptr.is_null() {
        write_err(err_buf, err_buf_len, "Null vector pointer");
        return -1;
    }

    let idx = match index_manager::get_index(name) {
        Ok(idx) => idx,
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            return -1;
        }
    };

    if dimension as usize != idx.dimension() {
        write_err(
            err_buf,
            err_buf_len,
            &format!(
                "Dimension mismatch: vector has {} but index expects {}",
                dimension,
                idx.dimension()
            ),
        );
        return -1;
    }

    let vector = std::slice::from_raw_parts(vector_ptr, dimension as usize);

    match idx.add(vector) {
        Ok(label) => label as i64,
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            -1
        }
    }
}

/// Create index: returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_create_index_buf(
    name: *const c_char,
    dimension: i32,
    metric: *const c_char,
    max_degree: i32,
    build_complexity: i32,
    alpha: f32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    let name = match cstr_to_str(name, "name", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };
    let metric_str = match cstr_to_str(metric, "metric", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };

    if dimension <= 0 {
        write_err(
            err_buf,
            err_buf_len,
            &format!("Invalid dimension: {} (must be > 0)", dimension),
        );
        return -1;
    }

    let m = match metric_str.to_lowercase().as_str() {
        "l2" => Metric::L2,
        "ip" | "inner_product" => Metric::InnerProduct,
        other => {
            write_err(
                err_buf,
                err_buf_len,
                &format!("Unknown metric '{}'. Supported: L2, IP", other),
            );
            return -1;
        }
    };

    match index_manager::create_index(
        name,
        dimension as usize,
        m,
        max_degree as u32,
        build_complexity as u32,
        alpha,
    ) {
        Ok(()) => 0,
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            -1
        }
    }
}

/// Destroy index: returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_destroy_index_buf(
    name: *const c_char,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    let name = match cstr_to_str(name, "name", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };
    match index_manager::destroy_index(name) {
        Ok(()) => 0,
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            -1
        }
    }
}

/// Save index: returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_save_index_buf(
    name: *const c_char,
    path: *const c_char,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    let name = match cstr_to_str(name, "name", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };
    let path = match cstr_to_str(path, "path", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };

    match index_manager::save_index(name, path) {
        Ok(()) => 0,
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            -1
        }
    }
}

/// Load index: returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_load_index_buf(
    name: *const c_char,
    path: *const c_char,
    build_complexity: i32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    let name = match cstr_to_str(name, "name", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };
    let path = match cstr_to_str(path, "path", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };

    let bc = if build_complexity > 0 {
        build_complexity as u32
    } else {
        0
    };

    match index_manager::load_index(name, path, bc) {
        Ok(()) => 0,
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            -1
        }
    }
}

/// Streaming build: two-pass external-memory index build from binary vectors file.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_streaming_build_buf(
    input_path: *const c_char,
    output_path: *const c_char,
    metric: *const c_char,
    max_degree: i32,
    build_complexity: i32,
    alpha: f32,
    sample_size: i32,
    out_num_vectors: *mut i32,
    out_dimension: *mut i32,
    out_sample_size: *mut i32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    let input = match cstr_to_str(input_path, "input_path", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };
    let output = match cstr_to_str(output_path, "output_path", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };
    let metric_str = match cstr_to_str(metric, "metric", err_buf, err_buf_len) {
        Some(s) => s,
        None => return -1,
    };

    let m = match metric_str.to_lowercase().as_str() {
        "l2" => Metric::L2,
        "ip" | "inner_product" => Metric::InnerProduct,
        other => {
            write_err(err_buf, err_buf_len, &format!("Unknown metric '{}'. Supported: L2, IP", other));
            return -1;
        }
    };

    let ss = if sample_size > 0 {
        sample_size as u32
    } else {
        // Default: sqrt(N), but we don't know N yet. Use 0 as sentinel.
        0
    };

    match crate::streaming_build::streaming_build(
        input,
        output,
        m,
        max_degree as u32,
        build_complexity as u32,
        alpha,
        ss,
    ) {
        Ok(result) => {
            if !out_num_vectors.is_null() {
                *out_num_vectors = result.num_vectors as i32;
            }
            if !out_dimension.is_null() {
                *out_dimension = result.dimension as i32;
            }
            if !out_sample_size.is_null() {
                *out_sample_size = result.sample_size as i32;
            }
            0
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            -1
        }
    }
}

// ========================================
// JSON-based functions (kept for list/info)
// ========================================

/// List all indexes. Returns JSON array of index info objects.
#[no_mangle]
pub extern "C" fn diskann_list_indexes() -> DiskannResult {
    let infos = index_manager::list_indexes();
    let items: Vec<String> = infos
        .iter()
        .map(|i| {
            format!(
                "{{\"name\":\"{}\",\"dimension\":{},\"count\":{},\"metric\":\"{}\",\"max_degree\":{},\"read_only\":{}}}",
                i.name, i.dimension, i.count, i.metric, i.max_degree, i.read_only
            )
        })
        .collect();
    ok_result(format!("[{}]", items.join(",")))
}

/// Get info for a single index. Returns JSON key-value object.
#[no_mangle]
pub unsafe extern "C" fn diskann_get_info(name: *const c_char) -> DiskannResult {
    let name = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(e) => return err_result(format!("Invalid name: {}", e)),
    };
    match index_manager::get_info(name) {
        Ok(info) => ok_result(format!(
            "{{\"name\":\"{}\",\"dimension\":{},\"count\":{},\"metric\":\"{}\",\"max_degree\":{},\"build_complexity\":{},\"alpha\":{},\"read_only\":{}}}",
            info.name, info.dimension, info.count, info.metric, info.max_degree, info.build_complexity, info.alpha, info.read_only
        )),
        Err(e) => err_result(e.to_string()),
    }
}

/// Check if an index exists. Returns 1 if exists, 0 if not.
#[no_mangle]
pub unsafe extern "C" fn diskann_index_exists(name: *const c_char) -> i32 {
    let name = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    if index_manager::get_index(name).is_ok() { 1 } else { 0 }
}

/// Get library version.
#[no_mangle]
pub extern "C" fn diskann_rust_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

// ========================================
// Detached handle FFI (for DuckDB BoundIndex)
// ========================================

/// Opaque handle to an InMemoryIndex not in the global registry.
pub type DiskannHandle = *mut InMemoryIndex;

/// FFI-safe serialized bytes.
#[repr(C)]
pub struct DiskannBytes {
    pub data: *mut u8,
    pub len: usize,
}

/// Create a detached index. Returns handle, or null on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_create_detached(
    dimension: i32,
    metric: *const c_char,
    max_degree: i32,
    build_complexity: i32,
    alpha: f32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> DiskannHandle {
    let metric_str = match cstr_to_str(metric, "metric", err_buf, err_buf_len) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };
    let m = match metric_str.to_lowercase().as_str() {
        "l2" => Metric::L2,
        "ip" | "inner_product" => Metric::InnerProduct,
        other => {
            write_err(
                err_buf,
                err_buf_len,
                &format!("Unknown metric '{}'", other),
            );
            return ptr::null_mut();
        }
    };
    let index = InMemoryIndex::new_detached(
        dimension as usize,
        m,
        max_degree as u32,
        build_complexity as u32,
        alpha,
    );
    Box::into_raw(Box::new(index))
}

/// Free a detached index handle.
#[no_mangle]
pub unsafe extern "C" fn diskann_free_detached(handle: DiskannHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Add a vector to a detached index. Returns assigned label, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_detached_add(
    handle: DiskannHandle,
    vector_ptr: *const f32,
    dimension: i32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i64 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "Null handle");
        return -1;
    }
    if vector_ptr.is_null() || dimension <= 0 {
        write_err(err_buf, err_buf_len, "Invalid vector");
        return -1;
    }
    let index = &*handle;
    let vector = std::slice::from_raw_parts(vector_ptr, dimension as usize);
    match index.add(vector) {
        Ok(label) => label as i64,
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            -1
        }
    }
}

/// Search a detached index. Returns number of results, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_detached_search(
    handle: DiskannHandle,
    query_ptr: *const f32,
    dimension: i32,
    k: i32,
    search_complexity: i32,
    out_labels: *mut i64,
    out_distances: *mut f32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "Null handle");
        return -1;
    }
    if query_ptr.is_null() || dimension <= 0 {
        write_err(err_buf, err_buf_len, "Invalid query");
        return -1;
    }
    let index = &*handle;
    let query = std::slice::from_raw_parts(query_ptr, dimension as usize);
    match index.search(query, k as usize, search_complexity as u32) {
        Ok(results) => {
            let n = results.len().min(k as usize);
            for i in 0..n {
                *out_labels.add(i) = results[i].0 as i64;
                *out_distances.add(i) = results[i].1;
            }
            n as i32
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            -1
        }
    }
}

/// Get vector count in a detached index.
#[no_mangle]
pub unsafe extern "C" fn diskann_detached_count(handle: DiskannHandle) -> i64 {
    if handle.is_null() {
        return 0;
    }
    let index = &*handle;
    index.len() as i64
}

/// Serialize a detached index to bytes. Caller must free with diskann_free_bytes.
#[no_mangle]
pub unsafe extern "C" fn diskann_detached_serialize(
    handle: DiskannHandle,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> DiskannBytes {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "Null handle");
        return DiskannBytes {
            data: ptr::null_mut(),
            len: 0,
        };
    }
    let index = &*handle;
    match index.serialize_to_bytes() {
        Ok(mut bytes) => {
            let len = bytes.len();
            let ptr = bytes.as_mut_ptr();
            std::mem::forget(bytes);
            DiskannBytes { data: ptr, len }
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            DiskannBytes {
                data: ptr::null_mut(),
                len: 0,
            }
        }
    }
}

/// Deserialize bytes into a detached index. Returns handle, or null on error.
#[no_mangle]
pub unsafe extern "C" fn diskann_detached_deserialize(
    data: *const u8,
    len: usize,
    alpha: f32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> DiskannHandle {
    if data.is_null() || len == 0 {
        write_err(err_buf, err_buf_len, "Null or empty data");
        return ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(data, len);
    match InMemoryIndex::from_bytes(bytes, alpha) {
        Ok(index) => Box::into_raw(Box::new(index)),
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            ptr::null_mut()
        }
    }
}

/// Free serialized bytes returned by diskann_detached_serialize.
#[no_mangle]
pub unsafe extern "C" fn diskann_free_bytes(bytes: DiskannBytes) {
    if !bytes.data.is_null() && bytes.len > 0 {
        let _ = Vec::from_raw_parts(bytes.data, bytes.len, bytes.len);
    }
}

// ========================================
// Compact / vacuum
// ========================================

/// Result of a compact operation.
#[repr(C)]
pub struct DiskannCompactResult {
    /// New index handle (caller owns it). Null on error.
    pub new_handle: DiskannHandle,
    /// Array of (old_label, new_label) pairs. Caller must free with diskann_free_label_map.
    pub label_map: *mut u32,
    /// Number of pairs (entries = map_len * 2 u32s).
    pub map_len: usize,
}

/// Compact a detached index by rebuilding without deleted labels.
/// `deleted_labels` is an array of u32 labels to exclude.
/// Returns a new handle + label mapping. Old handle is NOT freed.
#[no_mangle]
pub unsafe extern "C" fn diskann_detached_compact(
    handle: DiskannHandle,
    deleted_labels: *const u32,
    num_deleted: usize,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> DiskannCompactResult {
    let null_result = DiskannCompactResult {
        new_handle: ptr::null_mut(),
        label_map: ptr::null_mut(),
        map_len: 0,
    };

    if handle.is_null() {
        write_err(err_buf, err_buf_len, "Null handle");
        return null_result;
    }

    let index = &*handle;

    let mut deleted_set = std::collections::HashSet::with_capacity(num_deleted);
    if !deleted_labels.is_null() && num_deleted > 0 {
        let labels = std::slice::from_raw_parts(deleted_labels, num_deleted);
        for &l in labels {
            deleted_set.insert(l);
        }
    }

    match index.compact(&deleted_set) {
        Ok((new_index, label_map)) => {
            let new_handle = Box::into_raw(Box::new(new_index));

            // Flatten label_map to [old0, new0, old1, new1, ...]
            let map_len = label_map.len();
            let mut flat: Vec<u32> = Vec::with_capacity(map_len * 2);
            for (old_l, new_l) in &label_map {
                flat.push(*old_l);
                flat.push(*new_l);
            }
            let map_ptr = flat.as_mut_ptr();
            std::mem::forget(flat);

            DiskannCompactResult {
                new_handle,
                label_map: map_ptr,
                map_len,
            }
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            null_result
        }
    }
}

/// Free the label map returned by diskann_detached_compact.
#[no_mangle]
pub unsafe extern "C" fn diskann_free_label_map(map: *mut u32, map_len: usize) {
    if !map.is_null() && map_len > 0 {
        let _ = Vec::from_raw_parts(map, map_len * 2, map_len * 2);
    }
}

// ========================================
// Vector accessor (for MergeIndexes)
// ========================================

/// Get a copy of a vector by label. Returns dimension, or 0 if not found.
/// Caller provides output buffer `out_vec` of size >= dimension.
#[no_mangle]
pub unsafe extern "C" fn diskann_detached_get_vector(
    handle: DiskannHandle,
    label: u32,
    out_vec: *mut f32,
    out_capacity: i32,
) -> i32 {
    if handle.is_null() || out_vec.is_null() || out_capacity <= 0 {
        return 0;
    }
    let index = &*handle;
    match index.get_vector(label) {
        Some(v) => {
            let copy_len = v.len().min(out_capacity as usize);
            std::ptr::copy_nonoverlapping(v.as_ptr(), out_vec, copy_len);
            copy_len as i32
        }
        None => 0,
    }
}
