//! Probability distribution distance metrics: Hellinger, Jeffreys, and Jensen-Shannon distances.

use super::traits::Distance;

#[cfg(all(
    feature = "simdeez_f",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use super::disteez::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

///
/// A structure to compute Hellinger distance between probalilities.
/// Vector must be >= 0 and normalized to 1.
///   
/// The distance computation does not check that
/// and in fact simplifies the expression of distance assuming vectors are positive and L1 normalised to 1.
/// The user must enforce these conditions before  inserting otherwise results will be meaningless
/// at best or code will panic!
///
/// For f32 a simd implementation is provided if avx2 is detected.
#[derive(Default, Copy, Clone)]
pub struct DistHellinger;

// default implementation
macro_rules! implementHellingerDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistHellinger {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
            assert_eq!(va.len(), vb.len());
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
        // to // by rayon
            let mut dist = va.iter().zip(vb.iter()).map(|t| ((*t.0).sqrt() * (*t.1).sqrt()) as f32).fold(0., |acc , t| (acc + t*t));
            dist = (1. - dist).sqrt();
            dist
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementHellingerDistance!(f64);

impl Distance<f32> for DistHellinger {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        #[cfg(feature = "simdeez_f")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    return unsafe { distance_hellinger_f32_avx2(va, vb) };
                }
            }
        }
        let mut dist = va
            .iter()
            .zip(vb.iter())
            .map(|t| ((*t.0).sqrt() * (*t.1).sqrt()) as f32)
            .fold(0., |acc, t| acc + t);
        // if too far away from >= panic else reset!
        assert!(1. - dist >= -0.000001);
        dist = (1. - dist).max(0.).sqrt();
        dist
    } // end of eval
}

//=======================================================================================

///
/// A structure to compute Jeffreys divergence between probalilities.
/// If p and q are 2 probability distributions
/// the "distance" is computed as:
///   sum (p\[i\] - q\[i\]) * ln(p\[i\]/q\[i\])
///
/// To take care of null probabilities in the formula we use  max(x\[i\],1.E-30)
/// for x = p and q in the log compuations
///   
/// Vector must be >= 0 and normalized to 1!  
/// The distance computation does not check that.
/// The user must enforce these conditions before inserting in the hnws structure,
/// otherwise results will be meaningless at best or code will panic!
///
/// For f32 a simd implementation is provided if avx2 is detected.
#[derive(Default, Copy, Clone)]
pub struct DistJeffreys;

pub const M_MIN: f32 = 1.0e-30;

// default implementation
macro_rules! implementJeffreysDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistJeffreys {
        fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
        let dist = va.iter().zip(vb.iter()).map(|t| (*t.0 - *t.1) * ((*t.0).max(M_MIN as f64)/ (*t.1).max(M_MIN as f64)).ln() as f64).fold(0., |acc , t| (acc + t*t));
        dist as f32
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementJeffreysDistance!(f64);

impl Distance<f32> for DistJeffreys {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        //
        #[cfg(feature = "simdeez_f")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    return unsafe { distance_jeffreys_f32_avx2(va, vb) };
                }
            }
        }
        let dist = va
            .iter()
            .zip(vb.iter())
            .map(|t| (*t.0 - *t.1) * ((*t.0).max(M_MIN) / (*t.1).max(M_MIN)).ln() as f32)
            .fold(0., |acc, t| acc + t);
        dist
    } // end of eval
}

//=======================================================================================

/// Jensen-Shannon distance.  
/// It is defined as the **square root** of the  Jensen?Shannon divergence and is a metric.
/// Vector must be >= 0 and normalized to 1!
/// **The distance computation does not check that**.
#[derive(Default, Copy, Clone)]
pub struct DistJensenShannon;

macro_rules! implementDistJensenShannon (

    ($ty:ty) => (
        impl Distance<$ty> for DistJensenShannon {
            fn eval(&self, va:&[$ty], vb: &[$ty]) -> f32 {
                let mut dist = 0.;
                //
                assert_eq!(va.len(), vb.len());
                //
                for i in 0..va.len() {
                    let mean_ab = 0.5 * (va[i] + vb[i]);
                    if va[i] > 0. {
                        dist += va[i] * (va[i]/mean_ab).ln();
                    }
                    if vb[i] > 0. {
                        dist += vb[i] * (vb[i]/mean_ab).ln();
                    }
                }
                (0.5 * dist).sqrt() as f32
            } // end eval
        }  // end impl Distance<$ty>
    )  // end of pattern matching on ty
);

implementDistJensenShannon!(f64);
implementDistJensenShannon!(f32);
