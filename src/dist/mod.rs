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

// UniFrac distances
pub mod unifrac;
pub use unifrac::*;

// Public re-export module for compatibility
pub mod distances {
    // Re-export the Distance trait
    pub use super::traits::Distance;

    // Re-export all distance types
    pub use super::basic::*;
    pub use super::custom::*;
    pub use super::probability::*;
    pub use super::set::*;
    pub use super::string::*;
    pub use super::unifrac::*;

    // Test-only code
    #[cfg(test)]
    mod tests {
        use super::super::utils::l2_normalize;
        use super::*;

        // Include test code here if needed
    }
}
/// std simd distances
pub(crate) mod distsimd;

// simdeez distance implementation
pub(crate) mod disteez;
