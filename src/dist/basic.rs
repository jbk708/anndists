//! Basic vector distance metrics: L1, L2, Cosine, and Dot product distances.

use super::traits::Distance;

#[cfg(feature = "stdsimd")]
use super::distsimd::*;

#[cfg(feature = "simdeez_f")]
use super::disteez::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// L1 distance : implemented for i32, f64, i64, u32 , u16 , u8 and with Simd avx2 for f32
#[derive(Default, Copy, Clone)]
pub struct DistL1;

macro_rules! implementL1Distance (
    ($ty:ty) => (

    impl Distance<$ty> for DistL1  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
            va.iter().zip(vb.iter()).map(|t| (*t.0 as f32- *t.1 as f32).abs()).sum()
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementL1Distance!(i32);
implementL1Distance!(f64);
implementL1Distance!(i64);
implementL1Distance!(u32);
implementL1Distance!(u16);
implementL1Distance!(u8);

impl Distance<f32> for DistL1 {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        cfg_if::cfg_if! {
        if #[cfg(feature = "simdeez_f")] {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                if is_x86_feature_detected!("avx2") {
                    return unsafe {distance_l1_f32_avx2(va,vb)};
                }
                else {
                    assert_eq!(va.len(), vb.len());
                    va.iter().zip(vb.iter()).map(|t| (*t.0 as f32- *t.1 as f32).abs()).sum()
                }
            }
        }
        else if #[cfg(feature = "stdsimd")] {
            distance_l1_f32_simd(va,vb)
        }
        else {
            va.iter().zip(vb.iter()).map(|t| (*t.0 as f32- *t.1 as f32).abs()).sum()
        }
        } // end cfg_if
    } // end of eval
} // end impl Distance<f32> for DistL1

//========================================================================

/// L2 distance : implemented for i32, f64, i64, u32 , u16 , u8 and with Simd avx2 for f32
#[derive(Default, Copy, Clone)]
pub struct DistL2;

macro_rules! implementL2Distance (
    ($ty:ty) => (

    impl Distance<$ty> for DistL2  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
            let norm : f32 = va.iter().zip(vb.iter()).map(|t| (*t.0 as f32- *t.1 as f32) * (*t.0 as f32- *t.1 as f32)).sum();
            norm.sqrt()
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

//implementL2Distance!(f32);
implementL2Distance!(i32);
implementL2Distance!(f64);
implementL2Distance!(i64);
implementL2Distance!(u32);
implementL2Distance!(u16);
implementL2Distance!(u8);

#[allow(unused)]
// base scalar l2 for f32
fn scalar_l2_f32(va: &[f32], vb: &[f32]) -> f32 {
    let norm: f32 = va
        .iter()
        .zip(vb.iter())
        .map(|t| (*t.0 as f32 - *t.1 as f32) * (*t.0 as f32 - *t.1 as f32))
        .sum();
    assert!(norm >= 0.);
    return norm.sqrt();
}

impl Distance<f32> for DistL2 {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        cfg_if::cfg_if! {
            if #[cfg(feature = "simdeez_f")] {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        return unsafe { distance_l2_f32_avx2(va, vb) };
                    }
                    else {
                        return scalar_l2_f32(&va, &vb);
                    }
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    return scalar_l2_f32(&va, &vb);
                }
            } else if #[cfg(feature = "stdsimd")] {
                return distance_l2_f32_simd(va, vb);
            }
            else {
                return scalar_l2_f32(&va, &vb);
            }
        }
    } // end of eval
} // end impl Distance<f32> for DistL2

//=========================================================================

/// Cosine distance : implemented for f32, f64, i64, i32 , u16
#[derive(Default, Copy, Clone)]
pub struct DistCosine;

macro_rules! implementCosDistance(
    ($ty:ty) => (
     impl Distance<$ty> for DistCosine  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
            //
            let dist:f32;
            let zero:f64 = 0.;
            // to // by rayon
            let res = va.iter().zip(vb.iter()).map(|t| ((*t.0 * *t.1) as f64, (*t.0 * *t.0) as f64, (*t.1 * *t.1) as f64)).
                fold((0., 0., 0.), |acc , t| (acc.0 + t.0, acc.1 + t.1, acc.2 + t.2));
            //
            if res.1 > zero && res.2 > zero {
                let dist_unchecked = 1. - res.0 / (res.1 * res.2).sqrt();
                assert!(dist_unchecked >= - 0.00002);
                dist = dist_unchecked.max(0.) as f32;
            }
            else {
                dist = 0.;
            }
            //
            return dist;
        } // end of function
     } // end of impl block
    ) // end of matching
);

implementCosDistance!(f32);
implementCosDistance!(f64);
implementCosDistance!(i64);
implementCosDistance!(i32);
implementCosDistance!(u16);

//=========================================================================

/// This is essentially the Cosine distance but we suppose
/// all vectors (graph construction and request vectors have been l2 normalized to unity
/// BEFORE INSERTING in  HNSW!.   
/// No control is made, so it is the user responsability to send normalized vectors
/// everywhere in inserting and searching.
///
/// In large dimensions (hundreds) this pre-normalization spare cpu time.  
/// At low dimensions (a few ten's there is not a significant gain).  
/// This distance makes sense only for f16, f32 or f64
/// We provide for avx2 implementations for f32 that provides consequent gains
/// in large dimensions

#[derive(Default, Copy, Clone)]
pub struct DistDot;

#[allow(unused)]
macro_rules! implementDotDistance(
    ($ty:ty) => (
     impl Distance<$ty> for DistDot  {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
            //
            let zero:f32 = 0f32;
            // to // by rayon
            let dot = va.iter().zip(vb.iter()).map(|t| (*t.0 * *t.1) as f32).fold(0., |acc , t| (acc + t));
            //
            assert(dot <= 1.);
            return  1. - dot;
        } // end of function
      } // end of impl block
    ) // end of matching
);

#[allow(unused)]
fn scalar_dot_f32(va: &[f32], vb: &[f32]) -> f32 {
    let dot = 1.
        - va.iter()
            .zip(vb.iter())
            .map(|t| (*t.0 * *t.1) as f32)
            .fold(0., |acc, t| (acc + t));
    assert!(dot >= 0.);
    dot
}

impl Distance<f32> for DistDot {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        cfg_if::cfg_if! {
            if #[cfg(feature = "simdeez_f")] {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        return unsafe { distance_dot_f32_avx2(va, vb) };
                    } else if is_x86_feature_detected!("sse2") {
                        return unsafe { distance_dot_f32_sse2(va, vb) };
                    }
                    else {
                        return scalar_dot_f32(va, vb);
                    }
                } // end x86
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    return scalar_dot_f32(va, vb);
                }
            } else if #[cfg(feature = "stdsimd")] {
                return distance_dot_f32_simd_iter(va,vb);
            }
            else {
                return scalar_dot_f32(va, vb);
            }
        }
    } // end of eval
}
