//! C FFI interface for the DiskANN index manager.
//! Called from the C++ DuckDB extension.

use crate::index_manager::{self, Metric};
use std::ffi::{c_char, CStr, CString};
use std::ptr;

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

/// Create a new in-memory index.
/// metric: "L2" or "IP"
/// Returns JSON: {"status": "ok"}
#[no_mangle]
pub unsafe extern "C" fn diskann_create_index(
    name: *const c_char,
    dimension: i32,
    metric: *const c_char,
    max_degree: i32,
    build_complexity: i32,
) -> DiskannResult {
    let name = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(e) => return err_result(format!("Invalid name: {}", e)),
    };
    let metric_str = match CStr::from_ptr(metric).to_str() {
        Ok(s) => s,
        Err(e) => return err_result(format!("Invalid metric: {}", e)),
    };
    let m = match metric_str.to_lowercase().as_str() {
        "l2" => Metric::L2,
        "ip" | "inner_product" => Metric::InnerProduct,
        other => return err_result(format!("Unknown metric '{}'. Supported: L2, IP", other)),
    };

    match index_manager::create_index(
        name,
        dimension as usize,
        m,
        max_degree as u32,
        build_complexity as u32,
        1.2,
    ) {
        Ok(()) => ok_result(format!(
            "{{\"status\":\"Created index '{}' (dim={}, metric={}, R={}, L={})\"}}",
            name, dimension, metric_str, max_degree, build_complexity
        )),
        Err(e) => err_result(e.to_string()),
    }
}

/// Destroy an index.
#[no_mangle]
pub unsafe extern "C" fn diskann_destroy_index(name: *const c_char) -> DiskannResult {
    let name = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(e) => return err_result(format!("Invalid name: {}", e)),
    };
    match index_manager::destroy_index(name) {
        Ok(()) => ok_result(format!("{{\"status\":\"Destroyed index '{}'\" }}", name)),
        Err(e) => err_result(e.to_string()),
    }
}

/// Add a single vector. Returns JSON: {"label": 42}
/// vector_ptr points to `dimension` floats.
#[no_mangle]
pub unsafe extern "C" fn diskann_add_vector(
    name: *const c_char,
    vector_ptr: *const f32,
    dimension: i32,
) -> DiskannResult {
    let name = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(e) => return err_result(format!("Invalid name: {}", e)),
    };
    let vector = std::slice::from_raw_parts(vector_ptr, dimension as usize);

    let idx = match index_manager::get_index(name) {
        Ok(idx) => idx,
        Err(e) => return err_result(e.to_string()),
    };

    match idx.add(vector) {
        Ok(label) => ok_result(format!("{{\"label\":{}}}", label)),
        Err(e) => err_result(e.to_string()),
    }
}

/// Search for k nearest neighbors.
/// query_ptr points to `dimension` floats.
/// Returns JSON: {"results": [[label, distance], ...]}
#[no_mangle]
pub unsafe extern "C" fn diskann_search(
    name: *const c_char,
    query_ptr: *const f32,
    dimension: i32,
    k: i32,
) -> DiskannResult {
    let name = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(e) => return err_result(format!("Invalid name: {}", e)),
    };
    let query = std::slice::from_raw_parts(query_ptr, dimension as usize);

    let idx = match index_manager::get_index(name) {
        Ok(idx) => idx,
        Err(e) => return err_result(e.to_string()),
    };

    match idx.search(query, k as usize) {
        Ok(results) => {
            // Serialize as JSON array of [label, distance] pairs
            let pairs: Vec<String> = results
                .iter()
                .map(|(l, d)| format!("[{},{}]", l, d))
                .collect();
            ok_result(format!("{{\"results\":[{}]}}", pairs.join(",")))
        }
        Err(e) => err_result(e.to_string()),
    }
}

/// List all indexes. Returns JSON array of index info objects.
#[no_mangle]
pub extern "C" fn diskann_list_indexes() -> DiskannResult {
    let infos = index_manager::list_indexes();
    let items: Vec<String> = infos
        .iter()
        .map(|i| {
            format!(
                "{{\"name\":\"{}\",\"dimension\":{},\"count\":{},\"metric\":\"{}\",\"max_degree\":{}}}",
                i.name, i.dimension, i.count, i.metric, i.max_degree
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
            "{{\"name\":\"{}\",\"dimension\":{},\"count\":{},\"metric\":\"{}\",\"max_degree\":{}}}",
            info.name, info.dimension, info.count, info.metric, info.max_degree
        )),
        Err(e) => err_result(e.to_string()),
    }
}

/// Get library version.
#[no_mangle]
pub extern "C" fn diskann_rust_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}
