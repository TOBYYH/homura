//! Matrix products on ROCm (`csrc/gemm.hip`): the counterpart of the CPU backend's `gemm`, with
//! the same arguments.
//!
//! The library chooses a path for each description by fixed rules (no timing), so the same
//! description takes the same path, and gives the same bits, in every process. With
//! `HOMURA_ROCM_GEMM_LOG=1` the choice is printed to standard error once per description.

use std::collections::HashSet;
use std::ffi::CStr;
use std::sync::{Mutex, OnceLock};

use super::context::exec;
use super::copy::{buffer, device_index};
use super::{call, expect, ffi, workspace};
use crate::DType;
use crate::backend::Gemm;
use crate::backend::cpu::walk_strides;
use crate::tensor::TensorData;

/// The address of the element at the layout's offset (null for no memory).
fn address(data: &TensorData) -> *mut std::ffi::c_void {
    let b = buffer("gemm", data);
    // Without elements the layout may name an offset past an empty buffer; nothing is read.
    b.ptr()
        .wrapping_add(data.layout.offset * data.dtype.size_in_bytes())
        .cast()
}

/// The (batch, row, column) strides of a 3-D layout.
fn strides(data: &TensorData) -> [i64; 3] {
    let s = walk_strides(&data.layout);
    [s[0] as i64, s[1] as i64, s[2] as i64]
}

/// The description of `out = gemm(g)` for the library, folded by it (`hmr_gemm_fold`: batches
/// that share `b` become the rows of one batch), so that the selection, the log and the launch
/// all see the same description.
#[track_caller]
pub(super) fn describe(out: &TensorData, g: &Gemm) -> ffi::hmr_gemm_desc {
    let mut desc = unfolded(out, g);
    // SAFETY: a valid description; the call only rewrites its sizes and strides.
    expect(call(unsafe { ffi::hmr_gemm_fold(&mut desc) }));
    desc
}

/// The description of `out = gemm(g)` as the operands' layouts give it.
fn unfolded(out: &TensorData, g: &Gemm) -> ffi::hmr_gemm_desc {
    let (la, lb) = (&g.a.layout, &g.b.layout);
    let dtype = g.a.dtype;
    let float = dtype.is_floating_point();
    let reads_input = g.scale.as_ref().is_some_and(|s| s.beta.to_f64() != 0.0);
    let (c, c_strides) = match &g.scale {
        Some(s) if reads_input => (address(s.input).cast_const(), strides(s.input)),
        _ => (std::ptr::null(), [0; 3]),
    };
    let coefficient = |f: fn(&crate::backend::Scale) -> crate::Scalar| g.scale.as_ref().map(f);
    let (alpha, beta) = (coefficient(|s| s.alpha), coefficient(|s| s.beta));
    ffi::hmr_gemm_desc {
        dtype: dtype as i32,
        sum_batches: g.sum_batches as i32,
        scaled: g.scale.is_some() as i32,
        reads_input: reads_input as i32,
        batch: la.shape[0] as i64,
        m: la.shape[1] as i64,
        n: lb.shape[2] as i64,
        k: la.shape[2] as i64,
        a: address(g.a).cast_const(),
        b: address(g.b).cast_const(),
        c,
        out: address(out),
        a_strides: strides(g.a),
        b_strides: strides(g.b),
        c_strides,
        out_strides: strides(out),
        alpha: if float {
            alpha.map_or(0.0, |s| s.to_f64())
        } else {
            0.0
        },
        beta: if float {
            beta.map_or(0.0, |s| s.to_f64())
        } else {
            0.0
        },
        alpha_int: if float {
            0
        } else {
            alpha.map_or(0, |s| s.to_i64())
        },
        beta_int: if float {
            0
        } else {
            beta.map_or(0, |s| s.to_i64())
        },
    }
}

/// The path the library chooses for `desc`, and the layout of its factors.
#[track_caller]
pub(super) fn select(desc: &ffi::hmr_gemm_desc) -> (i32, i32) {
    let (mut path, mut layout) = (0, 0);
    // SAFETY: valid pointers; the call only reads the description's sizes, strides and
    // addresses.
    expect(call(unsafe {
        ffi::hmr_gemm_select(desc, &mut path, &mut layout)
    }));
    (path, layout)
}

/// The name of a path, as the selection rules and `HOMURA_ROCM_GEMM_LOG` call it.
pub(super) fn path_name(path: i32) -> String {
    let mut name = std::ptr::null();
    // SAFETY: a valid output location.
    expect(call(unsafe { ffi::hmr_gemm_path_name(path, &mut name) }));
    // SAFETY: the library returns a static NUL-terminated string.
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

fn layout_name(layout: i32) -> &'static str {
    match layout {
        ffi::HMR_GEMM_NN => "NN",
        ffi::HMR_GEMM_NT => "NT",
        ffi::HMR_GEMM_TN => "TN",
        ffi::HMR_GEMM_TT => "TT",
        _ => "strided",
    }
}

/// What the selection depends on: the description without its values and with the addresses
/// reduced to their alignment.
type Key = ([i32; 4], [i64; 4], [[i64; 3]; 4], [usize; 4], i32);

fn key(d: &ffi::hmr_gemm_desc, path: i32) -> Key {
    let align = |p: *const std::ffi::c_void| (p as usize) % 256;
    (
        [d.dtype, d.sum_batches, d.scaled, d.reads_input],
        [d.batch, d.m, d.n, d.k],
        [d.a_strides, d.b_strides, d.c_strides, d.out_strides],
        [
            align(d.a),
            align(d.b),
            align(d.c),
            align(d.out.cast_const()),
        ],
        path,
    )
}

/// Prints the choice for `desc` once, with `HOMURA_ROCM_GEMM_LOG=1`.
fn log(desc: &ffi::hmr_gemm_desc, dtype: DType, path: i32, layout: i32) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("HOMURA_ROCM_GEMM_LOG").as_deref() == Ok("1")) {
        return;
    }
    static SEEN: Mutex<Option<HashSet<Key>>> = Mutex::new(None);
    let new = SEEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashSet::new)
        .insert(key(desc, path));
    if new {
        let d = desc;
        eprintln!(
            "homura: gemm {dtype} {} batch={} m={} n={} k={} sum_batches={} scaled={} -> {}",
            layout_name(layout),
            d.batch,
            d.m,
            d.n,
            d.k,
            d.sum_batches,
            d.scaled,
            path_name(path)
        );
    }
}

/// `out = gemm(g)`; F16 and BF16 are rejected before (`backend::gemm`).
#[track_caller]
pub(crate) fn gemm(out: &TensorData, g: &Gemm) {
    let dtype = g.a.dtype;
    if dtype == DType::Bool {
        panic!("matmul is not supported for {dtype} tensors");
    }
    if out.layout.num_elements() == 0 {
        return;
    }
    let desc = describe(out, g);
    let (path, layout) = select(&desc);
    log(&desc, dtype, path, layout);
    let mut bytes = 0;
    // SAFETY: a valid description; the call only reads its sizes, strides and addresses.
    expect(call(unsafe {
        ffi::hmr_gemm_workspace(&desc, path, &mut bytes)
    }));
    let ws = workspace(bytes, out.device());
    let (ws_ptr, ws_bytes) = match &ws {
        Some(w) => (address(&w.data()), bytes),
        None => (std::ptr::null_mut(), 0),
    };
    let ctx = buffer("gemm", out).context();
    expect(exec(device_index(out), || {
        // SAFETY: the description's addresses are the operands' device memory, alive until
        // the call returns (the work is queued in stream order before any of it is freed).
        call(unsafe { ffi::hmr_gemm(ctx.raw(), &desc, path, ws_ptr, ws_bytes) })
    }));
}

/// Makes the selection on this thread choose `path` (`None`: its own rules), for tests.
#[cfg(test)]
pub(super) fn force_path(path: Option<i32>) {
    // SAFETY: sets a thread-local value.
    unsafe { ffi::hmr_test_force_gemm_path(path.unwrap_or(-1)) };
}
