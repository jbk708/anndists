//! Utility functions for distance computations.

/// Normalize a vector to unit L2 norm in-place.
///
/// This function modifies the input vector `va` so that its L2 norm becomes 1.
/// If the vector has zero norm, it remains unchanged.
///
/// # Arguments
/// * `va` - Mutable slice of f32 values to normalize
///
/// # Example
///
/// ```rust
/// use anndists::dist::utils::l2_normalize;
///
/// let mut v = vec![3.0, 4.0];
/// l2_normalize(&mut v);
/// assert!((v[0] - 0.6).abs() < 1e-6);
/// assert!((v[1] - 0.8).abs() < 1e-6);
/// ```
pub fn l2_normalize(va: &mut [f32]) {
    let l2norm = va.iter().map(|t| (*t * *t) as f32).sum::<f32>().sqrt();
    if l2norm > 0. {
        for i in 0..va.len() {
            va[i] = va[i] / l2norm;
        }
    }
}
