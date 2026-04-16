//! Aggregate function SDK for Daft extensions.
//!
//! Mirrors the structure of [`crate::function`] (scalar) but adds the
//! multi-phase lifecycle required for grouped aggregation:
//!
//! ```text
//! new_accumulator()          ← once per group
//!   │
//!   ├─ update(batch…)        ← one or more times per group (sequential worker)
//!   │
//!   ├─ merge(partial…)       ← tree-reduce across parallel workers
//!   │
//!   └─ finalize()            → single output value
//! ```
//!
//! `merge` is the key operation that enables distributed execution: each
//! worker builds a partial accumulator locally, then the host merges them
//! tree-wise before calling `finalize` exactly once per group.

use std::{
    any::Any,
    ffi::{CStr, c_char, c_int, c_void},
    sync::Arc,
};

use crate::{
    abi::{ArrowArray, ArrowData, ArrowSchema, FFI_AggFunction},
    error::{DaftError, DaftResult},
    ffi::trampoline::trampoline,
};

// ── Public traits ─────────────────────────────────────────────────────────────

/// Mutable state for a single group in a grouped aggregation.
///
/// The host creates one `Accumulator` per group via
/// [`DaftAggFunction::new_accumulator`], feeds row batches into it with
/// [`update`], optionally merges partial states from parallel workers with
/// [`merge`], and finally calls [`finalize`] to produce one output value.
///
/// Implementors must also provide [`as_any`] so that `merge` can downcast
/// `other` to the concrete type and access its internal state.
pub trait Accumulator: Send {
    /// Ingest one Arrow batch into this accumulator.
    ///
    /// `args` carries the same columns (in the same order) as the arguments
    /// passed to the aggregate function in the query.
    fn update(&mut self, args: Vec<ArrowData>) -> DaftResult<()>;

    /// Merge `other` (a partial accumulator from another worker) into `self`.
    ///
    /// After this call `other` is consumed and must not be used.
    /// Both `self` and `other` were created by the same
    /// [`DaftAggFunction::new_accumulator`] call, so their concrete types
    /// are guaranteed to match — use [`as_any`] to downcast.
    fn merge(&mut self, other: Box<dyn Accumulator>) -> DaftResult<()>;

    /// Produce the final single-value output for this group.
    ///
    /// The returned [`ArrowData`] must contain exactly one logical row.
    /// The accumulator is consumed after this call.
    fn finalize(self: Box<Self>) -> DaftResult<ArrowData>;

    /// Upcast to [`Any`] for type-safe downcasting inside [`merge`].
    ///
    /// Implement as: `fn as_any(self: Box<Self>) -> Box<dyn Any + Send> { self }`
    fn as_any(self: Box<Self>) -> Box<dyn Any + Send>;
}

/// Trait that extension authors implement to define an aggregate function.
///
/// # Example
///
/// ```rust,ignore
/// struct SumLengthFn;
///
/// impl DaftAggFunction for SumLengthFn {
///     fn name(&self) -> &CStr { c"sum_length" }
///
///     fn return_field(&self, _args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
///         // return Int64 field …
///     }
///
///     fn new_accumulator(&self) -> DaftResult<Box<dyn Accumulator>> {
///         Ok(Box::new(SumLengthAcc { total: 0 }))
///     }
/// }
/// ```
pub trait DaftAggFunction: Send + Sync {
    /// Unique function name exposed to Daft queries.
    fn name(&self) -> &CStr;

    /// Infer the output field type given the input field schemas.
    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema>;

    /// Allocate a fresh accumulator for a new group.
    fn new_accumulator(&self) -> DaftResult<Box<dyn Accumulator>>;
}

/// A shared, type-erased aggregate function reference.
pub type DaftAggFunctionRef = Arc<dyn DaftAggFunction>;

// ── FFI bridge internals ──────────────────────────────────────────────────────

/// Thin-pointer wrapper around a `Box<dyn Accumulator>`.
///
/// `Box<dyn Accumulator>` is a fat pointer (data + vtable = 2 words) and
/// cannot be passed as `*mut c_void` (1 word) directly.  We heap-allocate
/// the fat pointer so we can refer to it via a single thin pointer.
struct AccBox {
    // Option lets us `.take()` the accumulator in merge/finalize without
    // leaving a dangling reference inside the allocation.
    inner: Option<Box<dyn Accumulator>>,
}

// ── Public FFI constructor ────────────────────────────────────────────────────

/// Convert a [`DaftAggFunctionRef`] into a [`FFI_AggFunction`] vtable.
///
/// The `Arc` is moved into the vtable's opaque context and released
/// when the host calls `fini`.
pub fn agg_into_ffi(func: DaftAggFunctionRef) -> FFI_AggFunction {
    let ctx_ptr = Box::into_raw(Box::new(func));
    FFI_AggFunction {
        ctx: ctx_ptr.cast(),
        name: ffi_agg_name,
        return_field: ffi_agg_return_field,
        new_accumulator: ffi_new_accumulator,
        update: ffi_update,
        merge: ffi_merge,
        finalize: ffi_finalize,
        drop_accumulator: ffi_drop_accumulator,
        fini: ffi_agg_fini,
    }
}

// ── FFI callbacks ─────────────────────────────────────────────────────────────

unsafe extern "C" fn ffi_agg_name(ctx: *const c_void) -> *const c_char {
    unsafe { &*ctx.cast::<DaftAggFunctionRef>() }.name().as_ptr()
}

#[rustfmt::skip]
unsafe extern "C" fn ffi_agg_return_field(
    ctx:        *const c_void,
    args:       *const ArrowSchema,
    args_count: usize,
    ret:        *mut ArrowSchema,
    errmsg:     *mut *mut c_char,
) -> c_int {
    unsafe { trampoline(errmsg, "panic in agg return_field", || {
        let ctx = &*ctx.cast::<DaftAggFunctionRef>();
        let schemas = if args_count == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(args, args_count)
        };
        let result = ctx.return_field(schemas)?;
        std::ptr::write(ret, result);
        Ok(())
    })}
}

/// Allocate a new accumulator and return it as an opaque `*mut c_void`.
/// Returns null on error (with `*errmsg` set).
unsafe extern "C" fn ffi_new_accumulator(
    ctx:    *const c_void,
    errmsg: *mut *mut c_char,
) -> *mut c_void {
    let mut result_ptr: *mut c_void = std::ptr::null_mut();
    let rc = unsafe {
        trampoline(errmsg, "panic in new_accumulator", || {
            let ctx = &*ctx.cast::<DaftAggFunctionRef>();
            let acc = ctx.new_accumulator()?;
            let boxed = Box::new(AccBox { inner: Some(acc) });
            result_ptr = Box::into_raw(boxed).cast::<c_void>();
            Ok(())
        })
    };
    if rc != 0 { std::ptr::null_mut() } else { result_ptr }
}

#[rustfmt::skip]
unsafe extern "C" fn ffi_update(
    _ctx:        *const c_void,
    acc:         *mut c_void,
    args:        *const ArrowArray,
    args_schemas: *const ArrowSchema,
    args_count:  usize,
    errmsg:      *mut *mut c_char,
) -> c_int {
    unsafe { trampoline(errmsg, "panic in accumulator update", || {
        let acc_box = &mut *acc.cast::<AccBox>();
        let inner = acc_box.inner.as_mut().ok_or_else(|| {
            DaftError::RuntimeError("accumulator already finalized".into())
        })?;
        let mut data = Vec::with_capacity(args_count);
        for i in 0..args_count {
            let array  = std::ptr::read(args.add(i));
            let schema = std::ptr::read(args_schemas.add(i));
            data.push(ArrowData { schema, array });
        }
        inner.update(data)
    })}
}

#[rustfmt::skip]
unsafe extern "C" fn ffi_merge(
    _ctx:   *const c_void,
    dst:    *mut c_void,
    src:    *mut c_void,
    errmsg: *mut *mut c_char,
) -> c_int {
    unsafe { trampoline(errmsg, "panic in accumulator merge", || {
        // Take ownership of src's inner accumulator, then free the AccBox shell.
        let src_inner = (*src.cast::<AccBox>()).inner.take().ok_or_else(|| {
            DaftError::RuntimeError("src accumulator already consumed".into())
        })?;
        drop(Box::from_raw(src.cast::<AccBox>()));

        // Merge src into dst (dst keeps its AccBox alive).
        let dst_box = &mut *dst.cast::<AccBox>();
        dst_box.inner.as_mut()
            .ok_or_else(|| DaftError::RuntimeError("dst accumulator already finalized".into()))?
            .merge(src_inner)
    })}
}

#[rustfmt::skip]
unsafe extern "C" fn ffi_finalize(
    _ctx:       *const c_void,
    acc:        *mut c_void,
    ret_array:  *mut ArrowArray,
    ret_schema: *mut ArrowSchema,
    errmsg:     *mut *mut c_char,
) -> c_int {
    unsafe { trampoline(errmsg, "panic in accumulator finalize", || {
        // Reconstruct the Box so we own the allocation; take inner before drop.
        let mut acc_box = Box::from_raw(acc.cast::<AccBox>());
        let inner = acc_box.inner.take().ok_or_else(|| {
            DaftError::RuntimeError("accumulator already finalized".into())
        })?;
        // acc_box drops here (inner = None, allocation freed).
        let result = inner.finalize()?;
        std::ptr::write(ret_array, result.array);
        std::ptr::write(ret_schema, result.schema);
        Ok(())
    })}
}

/// Drop an accumulator without producing a result (e.g. on query cancellation).
unsafe extern "C" fn ffi_drop_accumulator(_ctx: *const c_void, acc: *mut c_void) {
    let _ = std::panic::catch_unwind(|| unsafe {
        drop(Box::from_raw(acc.cast::<AccBox>()));
    });
}

/// Free the function and all owned resources.
unsafe extern "C" fn ffi_agg_fini(ctx: *mut c_void) {
    let _ = std::panic::catch_unwind(|| unsafe {
        drop(Box::from_raw(ctx.cast::<DaftAggFunctionRef>()));
    });
}
