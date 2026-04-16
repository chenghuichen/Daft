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
pub trait ExtAggUDF: Send + Sync {
    fn name(&self) -> &str;
    fn get_return_field(&self, inputs: &[Field], schema: &Schema) -> DaftResult<Field>;
    fn call_agg(&self, inputs: Vec<Series>, groups: Option<&GroupIndices>) -> DaftResult<Series>;
}

/// A cloneable, hashable (by name) wrapper around `Arc<dyn ExtAggUDF>`.
///
/// `Hash` and `PartialEq` compare by function name only, matching the
/// expression-identity semantics used elsewhere in the planner.
#[derive(Clone)]
pub struct ExtAggHandle(pub Arc<dyn ExtAggUDF>);

impl ExtAggHandle {
    pub fn new(udf: Arc<dyn ExtAggUDF>) -> Self {
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

impl Hash for ExtAggHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.name().hash(state);
    }
}

impl PartialEq for ExtAggHandle {
    fn eq(&self, other: &Self) -> bool {
        self.0.name() == other.0.name()
    }
}

impl Eq for ExtAggHandle {}

impl std::fmt::Debug for ExtAggHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExtAggHandle({})", self.0.name())
    }
}

impl std::fmt::Display for ExtAggHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.name())
    }
}

impl Serialize for ExtAggHandle {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ExtAggHandle {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let udf = <Box<dyn ExtAggUDF>>::deserialize(deserializer)?;
        Ok(Self(Arc::from(udf)))
    }
}
