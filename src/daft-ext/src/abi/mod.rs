//! Stable C ABI contract between Daft and extension cdylibs.
//!
//! This module defines the `repr(C)` types that Daft and extension shared
//! libraries use to communicate. It has zero Daft internal dependencies
//! and zero Arrow implementation dependencies (unless a feature flag is enabled).
//!
//! Naming follows Postgres conventions:
//! - "module" = the shared library at the ABI boundary
//! - "extension" = the higher-level Python package wrapping a module

pub mod arrow;
pub mod compat;
pub mod ffi;

use std::ffi::{c_char, c_int, c_void};

pub use arrow::{ArrowArray, ArrowArrayStream, ArrowData, ArrowSchema};

/// Modules built against a different ABI version are rejected at load time.
///
/// History:
/// - v1: initial release (`FFI_ScalarFunction`, `FFI_SessionContext` with `define_function`)
/// - v2: added `FFI_AggFunction` and `define_agg_function` on `FFI_SessionContext`
pub const DAFT_ABI_VERSION: u32 = 2;

/// Symbol that every Daft module cdylib must export.
///
/// ```ignore
/// #[no_mangle]
/// pub extern "C" fn daft_module_magic() -> FFI_Module { ... }
/// ```
pub const DAFT_MODULE_MAGIC_SYMBOL: &str = "daft_module_magic";

/// Module definition returned by the entry point symbol.
///
/// Analogous to Postgres's `Pg_magic_struct` + `_PG_init` combined into
/// a single struct.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct FFI_Module {
    /// Must equal [`DAFT_ABI_VERSION`] or the loader rejects the module.
    pub daft_abi_version: u32,

    /// Module name as a null-terminated UTF-8 string.
    ///
    /// Must remain valid for the lifetime of the process (typically a
    /// `&'static CStr` cast to `*const c_char`).
    pub name: *const c_char,

    /// Called by the host to let the module register its functions.
    ///
    /// Returns 0 on success, non-zero on error.
    pub init: unsafe extern "C" fn(session: *mut FFI_SessionContext) -> c_int,

    /// Free a string previously allocated by this module
    /// (e.g. from `FFI_ScalarFunction::get_return_field` or error messages).
    pub free_string: unsafe extern "C" fn(s: *mut c_char),
}

// SAFETY: Function pointers plus a static string pointer.
unsafe impl Send for FFI_Module {}
unsafe impl Sync for FFI_Module {}

/// Virtual function table for a scalar function.
///
/// The host calls methods through these function pointers. `ctx` is an opaque
/// pointer owned by the module; the host never dereferences it directly.
#[repr(C)]
pub struct FFI_ScalarFunction {
    /// Opaque module-side context pointer.
    pub ctx: *const c_void,

    /// Return the function name as a null-terminated UTF-8 string.
    ///
    /// The returned pointer borrows from `ctx` and is valid until `fini`.
    pub name: unsafe extern "C" fn(ctx: *const c_void) -> *const c_char,

    /// Compute the output field given input fields.
    ///
    /// `args` points to `args_count` Arrow field schemas (C Data Interface).
    /// On success, writes the result schema to `*ret`.
    /// On error, writes a null-terminated message to `*errmsg`
    /// (freed by `FFI_Module::free_string`).
    ///
    /// Returns 0 on success, non-zero on error.
    pub get_return_field: unsafe extern "C" fn(
        ctx: *const c_void,
        args: *const ArrowSchema,
        args_count: usize,
        ret: *mut ArrowSchema,
        errmsg: *mut *mut c_char,
    ) -> c_int,

    /// Evaluate the function on Arrow arrays via the C Data Interface.
    ///
    /// On error, writes a null-terminated message to `*errmsg`
    /// (freed by `FFI_Module::free_string`).
    ///
    /// Returns 0 on success, non-zero on error.
    pub call: unsafe extern "C" fn(
        ctx: *const c_void,
        args: *const ArrowArray,
        args_schemas: *const ArrowSchema,
        args_count: usize,
        ret_array: *mut ArrowArray,
        ret_schema: *mut ArrowSchema,
        errmsg: *mut *mut c_char,
    ) -> c_int,

    /// Finalize the function, freeing all owned resources.
    pub fini: unsafe extern "C" fn(ctx: *mut c_void),
}

// SAFETY: The vtable is function pointers plus an opaque ctx pointer.
// The module is responsible for thread-safety of ctx.
unsafe impl Send for FFI_ScalarFunction {}
unsafe impl Sync for FFI_ScalarFunction {}

/// Virtual function table for an aggregate function.
///
/// Lifecycle (per group):
/// 1. `new_accumulator` — allocate fresh state; returns opaque `*mut c_void`
/// 2. `update` (one or more times) — feed a batch of rows into the accumulator
/// 3. `merge` (optional, for distributed/parallel execution) — merge a partial
///    accumulator (`src`) into another (`dst`); `src` is consumed by this call
/// 4. `finalize` — produce a single-value [`ArrowData`]; accumulator is consumed
///
/// If a group is abandoned (e.g. query cancelled), the host calls
/// `drop_accumulator` instead of `finalize`.
#[repr(C)]
pub struct FFI_AggFunction {
    /// Opaque module-side context pointer.
    pub ctx: *const c_void,

    /// Return the function name as a null-terminated UTF-8 string.
    pub name: unsafe extern "C" fn(ctx: *const c_void) -> *const c_char,

    /// Infer the output field type given input field schemas.
    /// Semantics identical to [`FFI_ScalarFunction::get_return_field`].
    pub return_field: unsafe extern "C" fn(
        ctx: *const c_void,
        args: *const ArrowSchema,
        args_count: usize,
        ret: *mut ArrowSchema,
        errmsg: *mut *mut c_char,
    ) -> c_int,

    /// Allocate a fresh accumulator for a new group.
    /// Returns an opaque `*mut c_void` on success, null on error (`*errmsg` set).
    pub new_accumulator:
        unsafe extern "C" fn(ctx: *const c_void, errmsg: *mut *mut c_char) -> *mut c_void,

    /// Feed one Arrow batch into an accumulator (borrows `acc`).
    pub update: unsafe extern "C" fn(
        ctx: *const c_void,
        acc: *mut c_void,
        args: *const ArrowArray,
        args_schemas: *const ArrowSchema,
        args_count: usize,
        errmsg: *mut *mut c_char,
    ) -> c_int,

    /// Merge `src` into `dst`.  `src` is consumed and freed by this call.
    pub merge: unsafe extern "C" fn(
        ctx: *const c_void,
        dst: *mut c_void,
        src: *mut c_void,
        errmsg: *mut *mut c_char,
    ) -> c_int,

    /// Produce the final single-row output; consumes and frees `acc`.
    pub finalize: unsafe extern "C" fn(
        ctx: *const c_void,
        acc: *mut c_void,
        ret_array: *mut ArrowArray,
        ret_schema: *mut ArrowSchema,
        errmsg: *mut *mut c_char,
    ) -> c_int,

    /// Free `acc` without producing output (query cancellation path).
    pub drop_accumulator: unsafe extern "C" fn(ctx: *const c_void, acc: *mut c_void),

    /// Finalize the function vtable, freeing all owned resources.
    pub fini: unsafe extern "C" fn(ctx: *mut c_void),
}

// SAFETY: The vtable is function pointers plus an opaque ctx pointer.
unsafe impl Send for FFI_AggFunction {}
unsafe impl Sync for FFI_AggFunction {}

/// Host-side session context passed to a module's `init` function.
///
/// The module calls `define_function` / `define_agg_function` to register
/// its scalar and aggregate functions respectively.
#[repr(C)]
pub struct FFI_SessionContext {
    /// Opaque host-side context pointer.
    pub ctx: *mut c_void,

    /// Register a scalar function with the host session.
    ///
    /// The host takes ownership of `function` on success.
    /// Returns 0 on success, non-zero on error.
    pub define_function:
        unsafe extern "C" fn(ctx: *mut c_void, function: FFI_ScalarFunction) -> c_int,

    /// Register an aggregate function with the host session.
    ///
    /// The host takes ownership of `function` on success.
    /// Returns 0 on success, non-zero on error.
    pub define_agg_function:
        unsafe extern "C" fn(ctx: *mut c_void, function: FFI_AggFunction) -> c_int,
}

// SAFETY: Function pointer plus opaque host pointer.
unsafe impl Send for FFI_SessionContext {}
unsafe impl Sync for FFI_SessionContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes() {
        let ptr = std::mem::size_of::<usize>();

        // FFI_ScalarFunction: ctx + name + get_return_field + call + fini = 5 pointers
        assert_eq!(std::mem::size_of::<FFI_ScalarFunction>(), 5 * ptr);

        // FFI_AggFunction: ctx + name + return_field + new_accumulator + update
        //                  + merge + finalize + drop_accumulator + fini = 9 pointers
        assert_eq!(std::mem::size_of::<FFI_AggFunction>(), 9 * ptr);

        // FFI_SessionContext: ctx + define_function + define_agg_function = 3 pointers
        assert_eq!(std::mem::size_of::<FFI_SessionContext>(), 3 * ptr);

        // FFI_Module: u32 (padded) + name + init + free_string
        // 64-bit: 4 + 4 pad + 8 + 8 + 8 = 32
        // 32-bit: 4 + 4 + 4 + 4 = 16
        assert_eq!(
            std::mem::size_of::<FFI_Module>(),
            if ptr == 8 { 32 } else { 16 }
        );
    }

    #[test]
    fn send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FFI_ScalarFunction>();
        assert_send_sync::<FFI_SessionContext>();
        assert_send_sync::<FFI_Module>();
    }

    #[test]
    fn constants() {
        // !! THIS TEST EXISTS SO THAT THESE ARE NOT CHANGED BY ACCIDENT
        // IT MEANS WE HAVE TO MANUALLY UPDATE IN TWO PLACES !!
        assert_eq!(DAFT_ABI_VERSION, 2);
        assert_eq!(DAFT_MODULE_MAGIC_SYMBOL, "daft_module_magic");
    }
}
