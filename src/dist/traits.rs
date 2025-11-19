//! Core trait for distance computation.
//!
//! This module defines the `Distance` trait which is the foundation for all
//! distance implementations in this crate.

/// This is the basic Trait describing a distance. The structure Hnsw can be instantiated by anything
/// satisfying this Trait. The crate provides implmentations for L1, L2 , Cosine, Jaccard, Hamming.
/// For other distances implement the trait possibly with the newtype pattern
///
/// # Example
///
/// ```rust,no_run
/// use anndists::dist::Distance;
///
/// pub struct DistL1;
///
/// impl Distance<f32> for DistL1 {
///     fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
///         va.iter().zip(vb.iter())
///             .map(|(a, b)| (a - b).abs())
///             .sum()
///     }
/// }
/// ```
pub trait Distance<T: Send + Sync> {
    /// Compute the distance between two vectors `va` and `vb`.
    ///
    /// # Arguments
    /// * `va` - First vector
    /// * `vb` - Second vector
    ///
    /// # Panics
    /// May panic if `va.len() != vb.len()` (implementation dependent)
    fn eval(&self, va: &[T], vb: &[T]) -> f32;
}
