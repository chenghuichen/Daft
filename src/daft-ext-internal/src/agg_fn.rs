use std::{
    ffi::{CStr, c_char},
    sync::Arc,
};

use common_error::{DaftError, DaftResult};
use daft_core::{
    array::ops::GroupIndices,
    prelude::{Field, Schema, Series, UInt64Array},
};
use daft_dsl::functions::AggFn;
use daft_ext::abi::{ArrowArray, ArrowSchema, FFI_AggFunction};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::module::ModuleHandle;

/// Shared state bundling an `FFI_AggFunction` vtable with its parent module.
struct AggInner {
    ffi: FFI_AggFunction,
    module: Arc<ModuleHandle>,
}

impl AggInner {
    /// Read and free an FFI-allocated C string.
    unsafe fn take_ffi_string(&self, ptr: *mut c_char) -> String {
        let s = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        unsafe { self.module.free_string(ptr) };
        s
    }

    fn check(&self, rc: i32, errmsg: *mut c_char, default_msg: &str) -> DaftResult<()> {
        if rc == 0 {
            return Ok(());
        }
        let msg = if errmsg.is_null() {
            default_msg.to_string()
        } else {
            unsafe { self.take_ffi_string(errmsg) }
        };
        Err(DaftError::InternalError(msg))
    }
}

impl Drop for AggInner {
    fn drop(&mut self) {
        unsafe { (self.ffi.fini)(self.ffi.ctx.cast_mut()) };
    }
}

unsafe impl Send for AggInner {}
unsafe impl Sync for AggInner {}

/// Host-side wrapper for `FFI_AggFunction`, implementing [`AggFn`].
///
/// This is the type that the session registry stores and that
/// `AggExpr::ExtensionAgg` carries inside an [`ExtAggHandle`].
pub struct AggFnHandle {
    name: &'static str,
    inner: Option<Arc<AggInner>>,
}

impl AggFnHandle {
    pub fn new(ffi: FFI_AggFunction, module: Arc<ModuleHandle>) -> Self {
        let name_ptr = unsafe { (ffi.name)(ffi.ctx) };
        let name: &'static str = Box::leak(
            unsafe { CStr::from_ptr(name_ptr) }
                .to_str()
                .expect("FFI agg function name must be valid UTF-8")
                .to_owned()
                .into_boxed_str(),
        );
        Self {
            name,
            inner: Some(Arc::new(AggInner { ffi, module })),
        }
    }

    fn inner(&self) -> DaftResult<&AggInner> {
        self.inner.as_deref().ok_or_else(|| {
            DaftError::InternalError(format!(
                "extension agg function '{}' is not loaded",
                self.name
            ))
        })
    }
}

/// Convert `Vec<Series>` to parallel FFI array + schema vecs.
fn series_to_ffi(series_vec: &[Series]) -> DaftResult<(Vec<ArrowArray>, Vec<ArrowSchema>)> {
    let mut ffi_arrays = Vec::with_capacity(series_vec.len());
    let mut ffi_schemas = Vec::with_capacity(series_vec.len());
    for s in series_vec {
        let arrow_field = s.field().to_arrow()?;
        let ffi_schema = arrow::ffi::FFI_ArrowSchema::try_from(arrow_field)
            .map_err(|e| DaftError::InternalError(format!("schema export failed: {e}")))?;
        let mut data = s.to_arrow()?.to_data();
        data.align_buffers();
        let ffi_array = arrow::ffi::FFI_ArrowArray::new(&data);
        ffi_arrays.push(unsafe { ArrowArray::from_owned(ffi_array) });
        ffi_schemas.push(unsafe { ArrowSchema::from_owned(ffi_schema) });
    }
    Ok((ffi_arrays, ffi_schemas))
}

/// Run a single accumulator lifecycle (new → update → finalize) on `inputs`.
/// Returns the raw Arrow array produced by `finalize`.
fn run_one_group(
    inner: &AggInner,
    inputs: Vec<Series>,
) -> DaftResult<arrow_array::ArrayRef> {
    let (ffi_arrays, ffi_schemas) = series_to_ffi(&inputs)?;

    // --- new_accumulator ---
    let mut errmsg: *mut c_char = std::ptr::null_mut();
    let acc = unsafe { (inner.ffi.new_accumulator)(inner.ffi.ctx, &raw mut errmsg) };
    if acc.is_null() {
        let msg = if errmsg.is_null() {
            "new_accumulator returned null".to_string()
        } else {
            unsafe { inner.take_ffi_string(errmsg) }
        };
        return Err(DaftError::InternalError(msg));
    }

    // --- update ---
    let mut errmsg: *mut c_char = std::ptr::null_mut();
    let rc = unsafe {
        (inner.ffi.update)(
            inner.ffi.ctx,
            acc,
            ffi_arrays.as_ptr(),
            ffi_schemas.as_ptr(),
            ffi_arrays.len(),
            &raw mut errmsg,
        )
    };
    if let Err(e) = inner.check(rc, errmsg, "update failed") {
        // Drop accumulator before propagating the error
        unsafe { (inner.ffi.drop_accumulator)(inner.ffi.ctx, acc) };
        return Err(e);
    }

    // --- finalize ---
    let mut ret_array = ArrowArray::empty();
    let mut ret_schema = ArrowSchema::empty();
    let mut errmsg: *mut c_char = std::ptr::null_mut();
    let rc = unsafe {
        (inner.ffi.finalize)(
            inner.ffi.ctx,
            acc,
            &raw mut ret_array,
            &raw mut ret_schema,
            &raw mut errmsg,
        )
    };
    inner.check(rc, errmsg, "finalize failed")?;

    let ffi_arr: arrow::ffi::FFI_ArrowArray = unsafe { ret_array.into_owned() };
    let ffi_sch: arrow::ffi::FFI_ArrowSchema = unsafe { ret_schema.into_owned() };
    let arrow_data = unsafe { arrow::ffi::from_ffi(ffi_arr, &ffi_sch) }
        .map_err(|e| DaftError::InternalError(format!("Arrow FFI import failed: {e}")))?;

    Ok(arrow_array::make_array(arrow_data))
}

#[typetag::serde(name = "AggFnHandle")]
impl AggFn for AggFnHandle {
    fn name(&self) -> &str {
        self.name
    }

    fn get_return_field(&self, inputs: &[Field], _schema: &Schema) -> DaftResult<Field> {
        let inner = self.inner()?;

        let ffi_schemas: Vec<ArrowSchema> = inputs
            .iter()
            .map(|f| {
                let arrow_field = f.to_arrow()?;
                arrow::ffi::FFI_ArrowSchema::try_from(arrow_field)
                    .map_err(|e| DaftError::InternalError(format!("schema export failed: {e}")))
                    .map(|ffi| unsafe { ArrowSchema::from_owned(ffi) })
            })
            .collect::<DaftResult<_>>()?;

        let mut ret_schema = ArrowSchema::empty();
        let mut errmsg: *mut c_char = std::ptr::null_mut();
        let rc = unsafe {
            (inner.ffi.return_field)(
                inner.ffi.ctx,
                ffi_schemas.as_ptr(),
                ffi_schemas.len(),
                &raw mut ret_schema,
                &raw mut errmsg,
            )
        };
        inner.check(rc, errmsg, "return_field failed")?;

        let ffi_schema: arrow::ffi::FFI_ArrowSchema = unsafe { ret_schema.into_owned() };
        let arrow_field = arrow_schema::Field::try_from(&ffi_schema)
            .map_err(|e| DaftError::InternalError(format!("schema import failed: {e}")))?;

        Field::try_from(&arrow_field)
    }

    fn call_agg(&self, inputs: Vec<Series>, groups: Option<&GroupIndices>) -> DaftResult<Series> {
        let inner = self.inner()?;

        // Resolve the return field upfront using the input field metadata.
        let input_fields: Vec<Field> = inputs
            .iter()
            .map(|s| Ok(s.field().clone()))
            .collect::<DaftResult<_>>()?;
        let ret_field = self.get_return_field(&input_fields, &Schema::empty())?;

        if let Some(groups) = groups {
            // One accumulator per group.
            let mut group_results: Vec<Series> = Vec::with_capacity(groups.len());

            for group_indices in groups {
                // Slice each input to just the rows in this group.
                let idx_arr = UInt64Array::from_values("", group_indices.iter().copied());
                let group_inputs: Vec<Series> = inputs
                    .iter()
                    .map(|s| s.take(&idx_arr))
                    .collect::<DaftResult<_>>()?;

                let arr = run_one_group(inner, group_inputs)?;
                let s = Series::from_arrow(ret_field.clone(), arr)?;
                group_results.push(s);
            }

            let refs: Vec<&Series> = group_results.iter().collect();
            Series::concat(&refs)
        } else {
            // Single accumulator for the whole partition.
            let arr = run_one_group(inner, inputs)?;
            Series::from_arrow(ret_field, arr)
        }
    }
}

impl Serialize for AggFnHandle {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("AggFnHandle", 1)?;
        s.serialize_field("name", self.name)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for AggFnHandle {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Helper {
            name: String,
        }
        let h = Helper::deserialize(deserializer)?;
        Ok(Self {
            name: Box::leak(h.name.into_boxed_str()),
            inner: None,
        })
    }
}

/// Create an [`AggFn`]-implementing [`AggFnHandle`] from an
/// `FFI_AggFunction` vtable.  Called during extension initialization.
pub fn into_agg_fn_handle(
    ffi: FFI_AggFunction,
    module: Arc<ModuleHandle>,
) -> Arc<dyn AggFn> {
    Arc::new(AggFnHandle::new(ffi, module))
}
