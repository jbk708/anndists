//! Set-based distance metrics: Hamming and Jaccard distances.

use super::traits::Distance;

#[cfg(feature = "stdsimd")]
use super::distsimd::*;

#[cfg(feature = "simdeez_f")]
use super::disteez::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Hamming distance. Implemented for u8, u16, u32, i32 and i16
/// The distance returned is normalized by length of slices, so it is between 0. and 1.  
///
/// A special implementation for f64 is made but exclusively dedicated to SuperMinHash usage in crate [probminhash](https://crates.io/crates/probminhash).  
/// It could be made generic with the PartialEq implementation for f64 and f32 in unsable source of Rust
#[derive(Default, Copy, Clone)]
pub struct DistHamming;

macro_rules! implementHammingDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistHamming  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
            assert_eq!(va.len(), vb.len());
            let norm : f32 = va.iter().zip(vb.iter()).filter(|t| t.0 != t.1).count() as f32;
            norm / va.len() as f32
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

impl Distance<i32> for DistHamming {
    fn eval(&self, va: &[i32], vb: &[i32]) -> f32 {
        //
        #[cfg(feature = "simdeez_f")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    return unsafe { distance_hamming_i32_avx2(va, vb) };
                }
            }
        }
        assert_eq!(va.len(), vb.len());
        let dist: f32 = va.iter().zip(vb.iter()).filter(|t| t.0 != t.1).count() as f32;
        dist / va.len() as f32
    } // end of eval
} // end implementation Distance<i32>

/// This implementation is dedicated to SuperMinHash algorithm in crate [probminhash](https://crates.io/crates/probminhash).  
/// Could be made generic with unstable source as there is implementation of PartialEq for f64
impl Distance<f64> for DistHamming {
    fn eval(&self, va: &[f64], vb: &[f64]) -> f32 {
        /*   Tests show that it is slower than basic method!!!
        #[cfg(feature = "simdeez_f")] {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                if is_x86_feature_detected!("avx2") {
                    log::trace!("calling distance_hamming_f64_avx2");
                    return unsafe { distance_hamming_f64_avx2(va,vb) };
                }
            }
        }
        */
        //
        assert_eq!(va.len(), vb.len());
        let dist: usize = va.iter().zip(vb.iter()).filter(|t| t.0 != t.1).count();
        (dist as f64 / va.len() as f64) as f32
    } // end of eval
} // end implementation Distance<f64>

//
/// This implementation is dedicated to SuperMinHash algorithm in crate [probminhash](https://crates.io/crates/probminhash).  
/// Could be made generic with unstable source as there is implementation of PartialEq for f32
impl Distance<f32> for DistHamming {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        cfg_if::cfg_if! {
            if #[cfg(feature = "stdsimd")] {
                return distance_jaccard_f32_16_simd(va,vb);
            }
            else {
                assert_eq!(va.len(), vb.len());
                let dist : usize = va.iter().zip(vb.iter()).filter(|t| t.0 != t.1).count();
                (dist as f64 / va.len() as f64) as f32
            }
        }
    } // end of eval
} // end implementation Distance<f32>

//
#[cfg(feature = "stdsimd")]
impl Distance<u32> for DistHamming {
    fn eval(&self, va: &[u32], vb: &[u32]) -> f32 {
        //
        return distance_jaccard_u32_16_simd(va, vb);
    } // end of eval
} // end implementation Distance<u32>

//
#[cfg(feature = "stdsimd")]
impl Distance<u64> for DistHamming {
    fn eval(&self, va: &[u64], vb: &[u64]) -> f32 {
        return distance_jaccard_u64_8_simd(va, vb);
    } // end of eval
} // end implementation Distance<u64>

// i32 is implmeented by simd
implementHammingDistance!(u8);
implementHammingDistance!(u16);

#[cfg(not(feature = "stdsimd"))]
implementHammingDistance!(u32);

#[cfg(not(feature = "stdsimd"))]
implementHammingDistance!(u64);

implementHammingDistance!(i16);

//====================================================================================
//   Jaccard Distance

/// Jaccard distance. Implemented for u8, u16 , u32.
#[derive(Default, Copy, Clone)]
pub struct DistJaccard;

// contruct a 2-uple accumulator that has sum of max in first component , and sum of min in 2 component
// stay in integer as long as possible
// Note : summing u32 coming from hash values can overflow! We must go up to u64 for additions!
macro_rules! implementJaccardDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistJaccard  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            let (max,min) : (u64, u64) = va.iter().zip(vb.iter()).fold((0u64,0u64), |acc, t| if t.0 > t.1 {
                                (acc.0 + *t.0 as u64, acc.1 + *t.1 as u64) }
                        else {
                                (acc.0 + *t.1 as u64 , acc.1 + *t.0 as u64)
                             }
            );
            if max > 0 {
                let dist = 1. - (min  as f64)/ (max as f64);
                assert!(dist >= 0.);
                dist as f32
            }
            else {
                0.
            }
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementJaccardDistance!(u8);
implementJaccardDistance!(u16);
implementJaccardDistance!(u32);
