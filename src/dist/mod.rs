//! module for distance implementation

// Core trait
pub mod traits;
pub use traits::Distance;

pub mod distances;
pub use distances::*;
/// std simd distances
pub(crate) mod distsimd;

// simdeez distance implementation
pub(crate) mod disteez;
