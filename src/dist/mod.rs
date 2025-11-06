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

// Set distances
pub mod set;
pub use set::*;

// String distances
pub mod string;
pub use string::*;

// Custom distances
pub mod custom;
pub use custom::*;

pub mod distances;
pub use distances::*;
/// std simd distances
pub(crate) mod distsimd;

// simdeez distance implementation
pub(crate) mod disteez;
