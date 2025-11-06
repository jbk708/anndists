//! module for distance implementation

// Core trait
pub mod traits;
pub use traits::Distance;

// Utilities
pub mod utils;

// Basic distances
pub mod basic;
pub use basic::*;

// Probability distances
pub mod probability;
pub use probability::*;

pub mod distances;
pub use distances::*;
/// std simd distances
pub(crate) mod distsimd;

// simdeez distance implementation
pub(crate) mod disteez;
