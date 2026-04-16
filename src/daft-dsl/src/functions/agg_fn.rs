use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use common_error::DaftResult;
use daft_core::{
    array::ops::GroupIndices,
    prelude::{Field, Schema, Series},
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Trait for native extension aggregate UDFs.
///
/// Implemented in `daft-ext-internal` by `AggFunctionHandle` and registered
/// with `typetag::serde` so that plan serialization/deserialization works.
///
/// Lifecycle per group:
/// 1. `call_agg` is the single entry point for both ungrouped and grouped
///    evaluation. Implementations create one accumulator per group internally
///    (or one accumulator for the ungrouped case).
#[typetag::serde(tag = "type")]
pub trait AggFn: Send + Sync {
    fn name(&self) -> &str;
    fn get_return_field(&self, inputs: &[Field], schema: &Schema) -> DaftResult<Field>;
    fn call_agg(&self, inputs: Vec<Series>, groups: Option<&GroupIndices>) -> DaftResult<Series>;
}

/// A cloneable, hashable (by name) wrapper around `Arc<dyn AggFn>`.
///
/// `Hash` and `PartialEq` compare by function name only, matching the
/// expression-identity semantics used elsewhere in the planner.
#[derive(Clone)]
pub struct AggFnHandle(pub Arc<dyn AggFn>);

impl AggFnHandle {
    pub fn new(udf: Arc<dyn AggFn>) -> Self {
        Self(udf)
    }

    pub fn name(&self) -> &str {
        self.0.name()
    }

    pub fn get_return_field(&self, inputs: &[Field], schema: &Schema) -> DaftResult<Field> {
        self.0.get_return_field(inputs, schema)
    }

    pub fn call_agg(
        &self,
        inputs: Vec<Series>,
        groups: Option<&GroupIndices>,
    ) -> DaftResult<Series> {
        self.0.call_agg(inputs, groups)
    }
}

impl Hash for AggFnHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.name().hash(state);
    }
}

impl PartialEq for AggFnHandle {
    fn eq(&self, other: &Self) -> bool {
        self.0.name() == other.0.name()
    }
}

impl Eq for AggFnHandle {}

impl std::fmt::Debug for AggFnHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AggFnHandle({})", self.0.name())
    }
}

impl std::fmt::Display for AggFnHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.name())
    }
}

impl Serialize for AggFnHandle {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AggFnHandle {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let udf = <Box<dyn AggFn>>::deserialize(deserializer)?;
        Ok(Self(Arc::from(udf)))
    }
}
