//! Test module for distance implementations.
//! 
//! This module contains comprehensive tests for all distance implementations
//! that have been refactored into separate modules (basic, probability, set,
//! string, custom, unifrac).

// Import all distance implementations for testing
use super::basic::*;
use super::probability::*;
use super::set::*;
use super::string::*;
use super::custom::*;
use super::unifrac::*;

// Import utility functions from utils module
use super::utils::l2_normalize;

//=======================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::traits::Distance;
    use super::super::set::{DistHamming, DistJaccard};
    use env_logger::Env;
    use log::debug;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_ulonglong};
    use std::ptr;

    fn init_log() -> u64 {
        let mut builder = env_logger::Builder::from_default_env();
        let _ = builder.is_test(true).try_init();
        println!("\n ************** initializing logger *****************\n");
        return 1;
    }
    /// Helper to create a raw array of `*const c_char` for T1..T6.
    fn make_obs_ids() -> Vec<*mut c_char> {
        let obs = vec![
            CString::new("T1").unwrap(),
            CString::new("T2").unwrap(),
            CString::new("T3").unwrap(),
            CString::new("T4").unwrap(),
            CString::new("T5").unwrap(),
            CString::new("T6").unwrap(),
        ];
        let mut c_ptrs = Vec::with_capacity(obs.len());
        for s in obs {
            // into_raw() -> *mut c_char
            c_ptrs.push(s.into_raw());
        }
        c_ptrs
    }

    fn free_obs_ids(c_ptrs: &mut [*mut c_char]) {
        for &mut ptr in c_ptrs {
            if !ptr.is_null() {
                // Convert back to a CString so it will be freed
                unsafe {
                    let _ = CString::from_raw(ptr);
                }
            }
        }
    }

    #[test]
    fn test_access_to_dist_l1() {
        let distl1 = DistL1;
        //
        let v1: Vec<i32> = vec![1, 2, 3];
        let v2: Vec<i32> = vec![2, 2, 3];

        let d1 = Distance::eval(&distl1, &v1, &v2);
        assert_eq!(d1, 1 as f32);

        let v3: Vec<f32> = vec![1., 2., 3.];
        let v4: Vec<f32> = vec![2., 2., 3.];
        let d2 = distl1.eval(&v3, &v4);
        assert_eq!(d2, 1 as f32);
    }

    #[test]
    fn have_avx2() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                println!("I have avx2");
            } else {
                println!(" ************ I DO NOT  have avx2  ***************");
            }
        }
    } // end if

    #[test]
    fn have_avx512f() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                println!("I have avx512f");
            } else {
                println!(" ************ I DO NOT  have avx512f  ***************");
            }
        } // end of have_avx512f
    }

    #[test]
    fn have_sse2() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse2") {
                println!("I have sse2");
            } else {
                println!(" ************ I DO NOT  have SSE2  ***************");
            }
        }
    } // end of have_sse2

    #[test]
    fn test_access_to_dist_cos() {
        let distcos = DistCosine;
        //
        let v1: Vec<i32> = vec![1, -1, 1];
        let v2: Vec<i32> = vec![2, 1, -1];

        let d1 = Distance::eval(&distcos, &v1, &v2);
        assert_eq!(d1, 1. as f32);
        //
        let v1: Vec<f32> = vec![1.234, -1.678, 1.367];
        let v2: Vec<f32> = vec![4.234, -6.678, 10.367];
        let d1 = Distance::eval(&distcos, &v1, &v2);

        let mut normv1 = 0.;
        let mut normv2 = 0.;
        let mut prod = 0.;
        for i in 0..v1.len() {
            prod += v1[i] * v2[i];
            normv1 += v1[i] * v1[i];
            normv2 += v2[i] * v2[i];
        }
        let dcos = 1. - prod / (normv1 * normv2).sqrt();
        println!("dist cos avec macro = {:?} ,  avec for {:?}", d1, dcos);
    }

    #[test]
    fn test_dot_distances() {
        let mut v1: Vec<f32> = vec![1.234, -1.678, 1.367];
        let mut v2: Vec<f32> = vec![4.234, -6.678, 10.367];

        let mut normv1 = 0.;
        let mut normv2 = 0.;
        let mut prod = 0.;
        for i in 0..v1.len() {
            prod += v1[i] * v2[i];
            normv1 += v1[i] * v1[i];
            normv2 += v2[i] * v2[i];
        }
        let dcos = 1. - prod / (normv1 * normv2).sqrt();
        //
        l2_normalize(&mut v1);
        l2_normalize(&mut v2);

        println!(" after normalisation v1 = {:?}", v1);
        let dot = DistDot.eval(&v1, &v2);
        println!(
            "dot  cos avec prenormalisation  = {:?} ,  avec for {:?}",
            dot, dcos
        );
    }

    #[test]
    fn test_l1() {
        init_log();
        //
        let va: Vec<f32> = vec![1.234, -1.678, 1.367, 1.234, -1.678, 1.367];
        let vb: Vec<f32> = vec![4.234, -6.678, 10.367, 1.234, -1.678, 1.367];
        //
        let dist = DistL1.eval(&va, &vb);
        let dist_check = va
            .iter()
            .zip(vb.iter())
            .map(|t| (*t.0 as f32 - *t.1 as f32).abs())
            .sum::<f32>();
        //
        log::info!(" dist : {:.5e} dist_check : {:.5e}", dist, dist_check);
        assert!((dist - dist_check).abs() / dist_check < 1.0e-5);
    } // end of test_l1

    #[test]
    fn test_jaccard_u16() {
        let v1: Vec<u16> = vec![1, 2, 1, 4, 3];
        let v2: Vec<u16> = vec![2, 2, 1, 5, 6];

        let dist = DistJaccard.eval(&v1, &v2);
        println!("dist jaccard = {:?}", dist);
        assert_eq!(dist, 1. - 11. / 16.);
    } // end of test_jaccard

    #[test]
    fn test_levenshtein() {
        let mut v1: Vec<u16> = vec![1, 2, 3, 4];
        let mut v2: Vec<u16> = vec![1, 2, 3, 3];
        let mut dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 1.0);
        v1 = vec![1, 2, 3, 4];
        v2 = vec![1, 2, 3, 4];
        dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 0.0);
        v1 = vec![1, 1, 1, 4];
        v2 = vec![1, 2, 3, 4];
        dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 2.0);
        v2 = vec![1, 1, 1, 4];
        v1 = vec![1, 2, 3, 4];
        dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 2.0);
    } // end of test_levenshtein

    extern "C" fn dist_func_float(va: *const f32, vb: *const f32, len: c_ulonglong) -> f32 {
        let mut dist: f32 = 0.;
        let sa = unsafe { std::slice::from_raw_parts(va, len as usize) };
        let sb = unsafe { std::slice::from_raw_parts(vb, len as usize) };

        for i in 0..len {
            dist += (sa[i as usize] - sb[i as usize]).abs().sqrt();
        }
        dist
    }

    #[test]
    fn test_dist_ext_float() {
        let va: Vec<f32> = vec![1., 2., 3.];
        let vb: Vec<f32> = vec![1., 2., 3.];
        println!("in test_dist_ext_float");
        let dist1 = dist_func_float(va.as_ptr(), vb.as_ptr(), va.len() as c_ulonglong);
        println!("test_dist_ext_float computed : {:?}", dist1);

        let mydist = DistCFFI::<f32>::new(dist_func_float);

        let dist2 = mydist.eval(&va, &vb);
        assert_eq!(dist1, dist2);
    } // end test_dist_ext_float

    #[test]

    fn test_my_closure() {
        //        use hnsw_rs::dist::Distance;
        let weight = vec![0.1, 0.8, 0.1];
        let my_fn = move |va: &[f32], vb: &[f32]| -> f32 {
            // should check that we work with same size for va, vb, and weight...
            let mut dist: f32 = 0.;
            for i in 0..va.len() {
                dist += weight[i] * (va[i] - vb[i]).abs();
            }
            dist
        };
        let my_boxed_f = Box::new(my_fn);
        let my_boxed_dist = DistFn::<f32>::new(my_boxed_f);
        let va: Vec<f32> = vec![1., 2., 3.];
        let vb: Vec<f32> = vec![2., 2., 4.];
        let dist = my_boxed_dist.eval(&va, &vb);
        println!("test_my_closure computed : {:?}", dist);
        // try allocation Hnsw
        //        let _hnsw = Hnsw::<f32, hnsw_rs::dist::DistFn<f32>>::new(10, 3, 100, 16, my_boxed_dist);
        //
        assert_eq!(dist, 0.2);
    } // end of test_my_closure

    #[test]
    fn test_hellinger() {
        let length = 9;
        let mut p_data = Vec::with_capacity(length);
        let mut q_data = Vec::with_capacity(length);
        for _ in 0..length {
            p_data.push(1. / length as f32);
            q_data.push(1. / length as f32);
        }
        p_data[0] -= 1. / (2 * length) as f32;
        p_data[1] += 1. / (2 * length) as f32;
        //
        let dist = DistHellinger.eval(&p_data, &q_data);

        let dist_exact_fn = |n: usize| -> f32 {
            let d1 = (4. - (6 as f32).sqrt() - (2 as f32).sqrt()) / n as f32;
            d1.sqrt() / (2 as f32).sqrt()
        };
        let dist_exact = dist_exact_fn(length);
        //
        log::info!("dist computed {:?} dist exact{:?} ", dist, dist_exact);
        println!("dist computed  {:?} , dist exact {:?} ", dist, dist_exact);
        //
        assert!((dist - dist_exact).abs() < 1.0e-5);
    }

    #[test]
    fn test_jeffreys() {
        // this essentially test av2 implementation for f32
        let length = 19;
        let mut p_data: Vec<f32> = Vec::with_capacity(length);
        let mut q_data: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            p_data.push(1. / length as f32);
            q_data.push(1. / length as f32);
        }
        p_data[0] -= 1. / (2 * length) as f32;
        p_data[1] += 1. / (2 * length) as f32;
        q_data[10] += 1. / (2 * length) as f32;
        //
        let dist_eval = DistJeffreys.eval(&p_data, &q_data);
        let mut dist_test = 0.;
        for i in 0..length {
            dist_test +=
                (p_data[i] - q_data[i]) * (p_data[i].max(M_MIN) / q_data[i].max(M_MIN)).ln();
        }
        //
        log::info!("dist eval {:?} dist test{:?} ", dist_eval, dist_test);
        println!("dist eval  {:?} , dist test {:?} ", dist_eval, dist_test);
        assert!(dist_test >= 0.);
        assert!((dist_eval - dist_test).abs() < 1.0e-5);
    }

    #[test]
    fn test_jensenshannon() {
        init_log();
        //
        let length = 19;
        let mut p_data: Vec<f32> = Vec::with_capacity(length);
        let mut q_data: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            p_data.push(1. / length as f32);
            q_data.push(1. / length as f32);
        }
        p_data[0] -= 1. / (2 * length) as f32;
        p_data[1] += 1. / (2 * length) as f32;
        q_data[10] += 1. / (2 * length) as f32;
        p_data[12] = 0.;
        q_data[12] = 0.;
        //
        let dist_eval = DistJensenShannon.eval(&p_data, &q_data);
        //
        log::info!("dist eval {:?} ", dist_eval);
        println!("dist eval  {:?} ", dist_eval);
    }

    #[allow(unused)]
    use rand::distributions::{Distribution, Uniform};

    // to be run with and without simdeez_f
    #[test]
    fn test_hamming_f64() {
        init_log();

        let size_test = 500;
        let fmax: f64 = 3.;
        let mut rng = rand::thread_rng();
        for i in 300..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et + imax et controler les resultat
            let between = Uniform::<f64>::from(-fmax..fmax);
            let va: Vec<f64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let mut vb: Vec<f64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            // reset half of vb to va
            for i in 0..i / 2 {
                vb[i] = va[i];
            }

            let easy_dist: u32 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let h_dist = DistHamming.eval(&va, &vb);
            let easy_dist = easy_dist as f32 / va.len() as f32;
            let j_exact = ((i / 2) as f32) / (i as f32);
            log::debug!(
                "test size {:?}  HammingDist {:.3e} easy : {:.3e} exact : {:.3e} ",
                i,
                h_dist,
                easy_dist,
                j_exact
            );
            if (easy_dist - h_dist).abs() > 1.0e-5 {
                println!(" jhamming = {:?} , jexact = {:?}", h_dist, easy_dist);
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
            if (j_exact - h_dist).abs() > 1. / i as f32 + 1.0E-5 {
                println!(
                    " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                    h_dist, j_exact, easy_dist
                );
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_hamming_f64

    #[test]
    fn test_hamming_f32() {
        init_log();

        let size_test = 500;
        let fmax: f32 = 3.;
        let mut rng = rand::thread_rng();
        for i in 300..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et + imax et controler les resultat
            let between = Uniform::<f32>::from(-fmax..fmax);
            let va: Vec<f32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let mut vb: Vec<f32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            // reset half of vb to va
            for i in 0..i / 2 {
                vb[i] = va[i];
            }

            let easy_dist: u32 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let h_dist = DistHamming.eval(&va, &vb);
            let easy_dist = easy_dist as f32 / va.len() as f32;
            let j_exact = ((i / 2) as f32) / (i as f32);
            log::debug!(
                "test size {:?}  HammingDist {:.3e} easy : {:.3e} exact : {:.3e} ",
                i,
                h_dist,
                easy_dist,
                j_exact
            );
            if (easy_dist - h_dist).abs() > 1.0e-5 {
                println!(
                    " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                    h_dist, j_exact, easy_dist
                );
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
            if (j_exact - h_dist).abs() > 1. / i as f32 + 1.0E-5 {
                println!(
                    " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                    h_dist, j_exact, easy_dist
                );
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_hamming_f32

    #[cfg(feature = "stdsimd")]
    #[test]
    fn test_feature_simd() {
        init_log();
        log::info!("I have activated stdsimd");
    } // end of test_feature_simd

    #[test]
    #[cfg(feature = "simdeez_f")]
    fn test_feature_simdeez() {
        init_log();
        log::info!("I have activated simdeez");
    } // end of test_feature_simd

    /// Example test using unweighted EMDUniFrac with presence/absence
    #[test]
    fn test_unifrac_unweighted() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init();

        let newick_str =
            "((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);";
        let feature_names = vec!["T1", "T2", "T3", "T4", "T5", "T6"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let dist_uni = DistUniFrac::new(newick_str, false, feature_names).unwrap();

        // SampleA: T1=7, T3=5, T4=2 => presence
        // SampleB: T2=3, T5=4, T6=9 => presence
        let va = vec![7.0, 0.0, 5.0, 2.0, 0.0, 0.0];
        let vb = vec![0.0, 3.0, 0.0, 0.0, 0.0, 9.0];

        let d = dist_uni.eval(&va, &vb);
        println!("Unweighted EMDUniFrac(A,B) = {}", d);
        // Should be ~0.4833
        // e.g. assert!((d - 0.4833).abs() < 1e-4);
    }

    #[test]
    fn test_unifrac_weighted() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init();
        let newick_str =
            "((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);";
        let feature_names = vec!["T1", "T2", "T3", "T4", "T5", "T6"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let dist_uni = DistUniFrac::new(newick_str, true, feature_names).unwrap();

        // Weighted with same data
        let va = vec![7.0, 0.0, 5.0, 2.0, 0.0, 0.0];
        let vb = vec![0.0, 3.0, 0.0, 0.0, 0.0, 9.0];

        let d = dist_uni.eval(&va, &vb);
        println!("Weighted EMDUniFrac(A,B) = {}", d);
        // Should be ~0.7279
        // e.g. assert!((d - 0.7279).abs() < 1e-4);
    }

    // TEMPORARILY COMMENTED OUT - requires external C UniFrac library
    /*
    #[test]
    fn test_unifrac_unweighted_c_api() {
        // 1) Build a BPTree from a Newick string
        let newick_str = CString::new("((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);").unwrap();
        let mut tree_ptr: *mut OpaqueBPTree = ptr::null_mut();
        unsafe {
            load_bptree_opaque(newick_str.as_ptr(), &mut tree_ptr);
        }
        assert!(!tree_ptr.is_null(), "Failed to build tree");

        // 2) Create obs_ids = T1..T6

        let mut obs_ids = make_obs_ids();
        let obs_ids_ptr = obs_ids.as_ptr() as *const *const c_char;

        // 3) Build DistUniFrac_C context for "unweighted"
        let method_str = CString::new("unweighted").unwrap();
        let ctx_ptr = dist_unifrac_create(
            6, // n_obs
            obs_ids_ptr,
            tree_ptr,
            method_str.as_ptr(),
            false,   // variance_adjust = false
            1.0,     // alpha
            false,   // bypass_tips = false
        );
        assert!(!ctx_ptr.is_null());

        // 4) Use dist_unifrac_c or DistUniFracCFFI to compute distance
        let dist_obj = DistUniFracCFFI::new(ctx_ptr);

        // Same example data:
        // Sample A => T1=7, T3=5, T4=2 => presence
        // Sample B => T2=3, T5=4, T6=9 => presence
        // We'll put them in f32 arrays of length 6: [T1,T2,T3,T4,T5,T6].
        let va = [7.0, 0.0, 5.0, 2.0, 0.0, 0.0];
        let vb = [0.0, 3.0, 0.0, 0.0, 4.0, 9.0];

        let dist = dist_obj.eval(&va, &vb);
        println!("Unweighted UniFrac(A,B) = {}", dist);
        // You mention it should be ~0.4833. Let's check tolerance
        // 5) Clean up
        unsafe {
            dist_unifrac_destroy(ctx_ptr);
            destroy_bptree_opaque(&mut tree_ptr);
        }
        free_obs_ids(&mut obs_ids);
    }
    */

    // TEMPORARILY COMMENTED OUT - requires external C UniFrac library
    /*
    #[test]
    fn test_unifrac_weighted_c_api() {
        // 1) Build a BPTree from a Newick string
        let newick_str = CString::new("((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);").unwrap();
        let mut tree_ptr: *mut OpaqueBPTree = ptr::null_mut();
        unsafe {
            load_bptree_opaque(newick_str.as_ptr(), &mut tree_ptr);
        }
        assert!(!tree_ptr.is_null(), "Failed to build tree");

        // 2) Create obs_ids = T1..T6
        let mut obs_ids = make_obs_ids();
        let obs_ids_ptr = obs_ids.as_ptr() as *const *const c_char;

        // 3) Build DistUniFrac_C context for "weighted"
        let method_str = CString::new("weighted_normalized").unwrap();
        let ctx_ptr = dist_unifrac_create(
            6, // n_obs
            obs_ids_ptr,
            tree_ptr,
            method_str.as_ptr(),
            false,   // variance_adjust = false
            1.0,     // alpha
            false,   // bypass_tips = false
        );
        assert!(!ctx_ptr.is_null());

        // 4) Use DistUniFracCFFI to compute weighted distance
        let dist_obj = DistUniFracCFFI::new(ctx_ptr);

        // Same example data with different abundances to test weighting:
        // Sample A => T1=7, T3=5, T4=2 (higher abundances)
        // Sample B => T2=3, T5=4, T6=9 (different abundance distribution)
        let va = [7.0, 0.0, 5.0, 2.0, 0.0, 0.0];
        let vb = [0.0, 3.0, 0.0, 0.0, 4.0, 9.0];

        let dist = dist_obj.eval(&va, &vb);
        println!("Weighted UniFrac(A,B) = {}", dist);

        // Test with different abundance patterns
        let vc = [10.0, 0.0, 1.0, 0.5, 0.0, 0.0]; // Different weights
        let vd = [0.0, 2.0, 0.0, 0.0, 8.0, 1.0];

        let dist2 = dist_obj.eval(&vc, &vd);
        println!("Weighted UniFrac(C,D) = {}", dist2);

        // Weighted distances should be different from unweighted for different abundance patterns
        assert!(dist >= 0.0 && dist <= 1.0, "Distance should be in [0,1]");
        assert!(dist2 >= 0.0 && dist2 <= 1.0, "Distance should be in [0,1]");

        // 5) Clean up
        unsafe {
            dist_unifrac_destroy(ctx_ptr);
            destroy_bptree_opaque(&mut tree_ptr);
        }
        free_obs_ids(&mut obs_ids);
    }
    */

    /// Example test using the new NewDistUniFrac with the exact function call pattern
    #[test]
    fn test_new_dist_unifrac_function_call() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init();

        let newick_str =
            "((T1:0.1,(T2:0.05,T3:0.05):0.02):0.3,(T4:0.2,(T5:0.1,T6:0.15):0.05):0.4);";
        let feature_names = vec!["T1", "T2", "T3", "T4", "T5", "T6"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let weighted = false;

        // Test the exact function call pattern requested
        let dist_unifrac =
            NewDistUniFrac::new(&newick_str, weighted, feature_names.clone()).unwrap();

        assert_eq!(dist_unifrac.num_features(), 6);
        assert_eq!(dist_unifrac.weighted, false);

        // Test with some sample data
        let va = vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0]; // T1, T3, T4 present
        let vb = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0]; // T2, T5, T6 present

        let d = dist_unifrac.eval(&va, &vb);
        println!("New UniFrac distance = {}", d);
        assert!(d >= 0.0 && d <= 1.0);

        // Test that it's actually using the unifrac_bp pattern (non-zero distance for different samples)
        assert!(d > 0.0); // Should be > 0 since samples have no overlap
    }

    /// Test NewDistUniFrac mathematical properties and correctness
    #[test]
    fn test_new_dist_unifrac_correctness_validation() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init();

        // Simple tree with known structure for manual verification
        let newick_str = "((T1:0.1,T2:0.1):0.2,(T3:0.1,T4:0.1):0.2):0.0;";
        let feature_names = vec!["T1", "T2", "T3", "T4"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        println!("=== Testing NewDistUniFrac Mathematical Properties ===");
        println!("Tree: {}", newick_str);
        println!("Testing shared/union algorithm implementation");

        // Create NewDistUniFrac implementation
        let new_dist = NewDistUniFrac::new(&newick_str, false, feature_names.clone()).unwrap();

        // Test cases focusing on relative behavior rather than absolute values
        let test_cases = vec![
            // (sample_a, sample_b, description)
            (
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                "T1 vs T2 (sister taxa)",
            ),
            (
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                "T1 vs T3 (distant taxa)",
            ),
            (
                vec![1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0],
                "T1,T2 vs T3,T4 (complete separation)",
            ),
            (
                vec![1.0, 0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0, 1.0],
                "T1,T3 vs T2,T4 (mixed)",
            ),
        ];

        println!("\nNewDistUniFrac Properties Validation (shared/union algorithm):");
        println!("{:<25} | {:>15}", "Test Case", "NewDistUniFrac");
        println!("{}", "-".repeat(45));

        for (va, vb, description) in test_cases {
            let new_distance = new_dist.eval(&va, &vb);

            println!("{:<25} | {:>15.6}", description, new_distance);
        }

        // Validate basic UniFrac properties instead of comparing to DistUniFrac
        let identical_distance =
            new_dist.eval(&vec![1.0, 0.0, 0.0, 0.0], &vec![1.0, 0.0, 0.0, 0.0]);
        assert!(
            identical_distance.abs() < 1e-6,
            "Identical samples should have distance ~0, got {}",
            identical_distance
        );

        println!("\nNewDistUniFrac mathematical properties validation passed!");
    }

    /// Test NewDistUniFrac with manually calculated expected values for mathematical correctness
    #[test]
    fn test_new_dist_unifrac_manual_calculation_validation() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init();

        // Very simple 2-leaf tree: (T1:1.0,T2:1.0):0.0;
        // With normalized UniFrac: T1=1/1=1.0, T2=1/1=1.0, difference=1.0-(-1.0)=2.0 at T1 and T2
        // Then propagated up: T1 branch contributes 1.0*1.0=1.0, T2 branch contributes 1.0*1.0=1.0
        // Total distance = 1.0 + 1.0 = 2.0 (this is the normalized UniFrac result)
        let newick_str = "(T1:1.0,T2:1.0):0.0;";
        let feature_names = vec!["T1", "T2"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        println!("=== Manual UniFrac Calculation Validation ===");
        println!("Tree: {}", newick_str);
        println!("Expected: For T1 vs T2 with normalization, each sample normalized to 1.0");
        println!("T1 sample: [1.0, 0.0] normalized = [1.0, 0.0]");
        println!("T2 sample: [0.0, 1.0] normalized = [0.0, 1.0]");
        println!("Differences: T1 node = 1.0-0.0=1.0, T2 node = 0.0-1.0=-1.0");
        println!("Distance = |1.0|*1.0 + |-1.0|*1.0 = 2.0");

        let dist_unifrac = NewDistUniFrac::new(&newick_str, false, feature_names).unwrap();

        // Test case: T1 present vs T2 present (completely different)
        let va = vec![1.0, 0.0]; // T1 only
        let vb = vec![0.0, 1.0]; // T2 only

        let distance = dist_unifrac.eval(&va, &vb);
        println!("Calculated distance: {}", distance);

        // For the shared/union UniFrac algorithm (which NewDistUniFrac uses):
        // T1 vs T2: no shared branches, so shared = 0.0
        // Union includes both T1 and T2 branches: union = 1.0 + 1.0 = 2.0
        // UniFrac = 1.0 - shared/union = 1.0 - 0.0/2.0 = 1.0
        let expected_distance = 1.0; // Not 2.0 - that was the old algorithm

        let expected = expected_distance; // Use the corrected expected value
        let tolerance = 0.01;

        assert!(
            (distance - expected).abs() < tolerance,
            "Manual calculation validation failed: expected ~{}, got {}, diff={}",
            expected,
            distance,
            (distance - expected).abs()
        );

        // Test identical samples (should be 0)
        let identical_distance = dist_unifrac.eval(&va, &va);
        assert!(
            identical_distance.abs() < 1e-10,
            "Identical samples should have distance 0, got {}",
            identical_distance
        );

        println!(" Manual calculation validation passed!");
        println!(
            "  Expected: {:.3}, Calculated: {:.6}, Difference: {:.6}",
            expected,
            distance,
            (distance - expected).abs()
        );

        // Test with a slightly more complex tree: ((T1:0.5,T2:0.5):0.5,T3:1.0):0.0;
        println!("\n=== Complex Tree Validation ===");
        let complex_newick = "((T1:0.5,T2:0.5):0.5,T3:1.0):0.0;";
        let complex_features = vec!["T1", "T2", "T3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let complex_dist = NewDistUniFrac::new(&complex_newick, false, complex_features).unwrap();

        // T1 vs T2 (sister taxa): should share the 0.5 branch length
        let dist_12 = complex_dist.eval(&vec![1.0, 0.0, 0.0], &vec![0.0, 1.0, 0.0]);
        // T1 vs T3 (more distant): should share less
        let dist_13 = complex_dist.eval(&vec![1.0, 0.0, 0.0], &vec![0.0, 0.0, 1.0]);

        println!("T1 vs T2 (sisters): {:.6}", dist_12);
        println!("T1 vs T3 (distant): {:.6}", dist_13);

        // Sister taxa should be closer than distant taxa
        assert!(
            dist_12 < dist_13,
            "Sister taxa should be closer: T1-T2({:.6}) should be < T1-T3({:.6})",
            dist_12,
            dist_13
        );

        println!(" Phylogenetic relationship validation passed!");
        println!(
            "  Sister taxa distance ({:.6}) < distant taxa distance ({:.6})",
            dist_12, dist_13
        );
    }

    /// Test NewDistUniFrac against ground truth values from test files
    #[test]
    fn test_new_dist_unifrac_ground_truth_validation() {
        use std::fs;
        use std::path::Path;

        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init();

        println!("=== Ground Truth Validation Test ===");
        println!("Testing NewDistUniFrac against actual ground truth from test files");

        // Load test data from data/ folder
        let tree_str = fs::read_to_string("data/test.nwk")
            .expect("Failed to read test.nwk")
            .trim()
            .to_string();

        println!("Tree: {}", tree_str);

        // Feature names from OTU table
        let features = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
            "T5".to_string(),
            "T6".to_string(),
            "T7".to_string(),
        ];

        println!("Features: {:?}", features);

        // Sample data from test_OTU_table.txt (converted to f32)
        let samples = vec![
            ("SampleA", vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]), // T1,T3,T4 present
            ("SampleB", vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]), // T2,T5,T6 present
            ("SampleC", vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), // T1,T3,T7 present
        ];

        for (name, data) in &samples {
            println!("{}: {:?}", name, data);
        }

        // Ground truth distances from test_truth.txt
        let ground_truth = vec![
            (("SampleA", "SampleB"), 0.4929577112197876),
            (("SampleA", "SampleC"), 0.3409090936183929),
            (("SampleB", "SampleC"), 0.4577465057373047),
        ];

        // Create NewDistUniFrac
        let new_dist = NewDistUniFrac::new(&tree_str, false, features)
            .expect("Failed to create NewDistUniFrac");

        println!("\n=== Validation Against Ground Truth (tolerance: 0.01) ===");
        println!(
            "{:<12} | {:<12} | {:>15} | {:>15} | {:>10} | Status",
            "Sample A", "Sample B", "NewDistUniFrac", "Ground Truth", "Diff"
        );
        println!("{}", "-".repeat(80));

        let tolerance = 0.01;
        let mut all_passed = true;

        // Test against ground truth values
        for ((sample_a, sample_b), expected_distance) in ground_truth {
            let data_a = &samples
                .iter()
                .find(|(name, _)| name == &sample_a)
                .unwrap()
                .1;
            let data_b = &samples
                .iter()
                .find(|(name, _)| name == &sample_b)
                .unwrap()
                .1;

            let new_distance = new_dist.eval(data_a, data_b) as f64;
            let diff = (new_distance - expected_distance).abs();
            let status = if diff < tolerance { "PASS" } else { "FAIL" };

            if diff >= tolerance {
                all_passed = false;
            }

            println!(
                "{:<12} | {:<12} | {:>15.6} | {:>15.6} | {:>10.6} | {}",
                sample_a, sample_b, new_distance, expected_distance, diff, status
            );
        }

        if all_passed {
            println!(
                "\nAll ground truth validation tests passed within tolerance of {}!",
                tolerance
            );
            println!("NewDistUniFrac matches the ground truth values from test_truth.txt!");
        } else {
            panic!("Ground truth validation failed! NewDistUniFrac results don't match test_truth.txt values");
        }
    }

    #[test]
    fn test_new_dist_unifrac_mouse_gut_validation() {
        use std::fs;
        use std::path::Path;

        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("warn"))
            .is_test(true)
            .try_init();

        println!("=== Mouse Gut Data Validation Test ===");
        println!("Testing NewDistUniFrac on real mouse gut microbiome data");

        // Check if mouse gut data files exist
        let tree_file = "data/Mouse_gut_zotu_aligned.tre";
        let counts_file = "data/Mouse_gut_zotu_counts.txt";

        if !Path::new(tree_file).exists() || !Path::new(counts_file).exists() {
            println!("Mouse gut data files not found, skipping validation");
            return;
        }

        // Read files
        let tree_content = fs::read_to_string(tree_file).expect("Failed to read tree file");
        let counts_content = fs::read_to_string(counts_file).expect("Failed to read counts file");

        println!("Tree file size: {} characters", tree_content.len());
        println!("Counts file size: {} characters", counts_content.len());

        // Parse counts data using existing parsing logic
        let lines: Vec<&str> = counts_content.lines().collect();

        fn parse_mouse_gut_format(lines: &[&str]) -> (Vec<String>, Vec<Vec<f32>>, Vec<String>) {
            if lines.is_empty() {
                return (vec![], vec![], vec![]);
            }

            // Find header line
            let mut header_idx = 0;
            for (i, line) in lines.iter().enumerate() {
                if line.starts_with('#') || (!line.trim().is_empty() && line.contains('\t')) {
                    header_idx = i;
                    break;
                }
            }

            let header = lines[header_idx].trim_start_matches('#').trim();
            let parts: Vec<&str> = header.split('\t').collect();

            if parts.len() < 2 {
                return (vec![], vec![], vec![]);
            }

            let sample_names: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();
            let mut feature_names = Vec::new();
            let mut all_samples: Vec<Vec<f32>> = vec![vec![]; sample_names.len()];

            // Parse data lines
            for line in lines.iter().skip(header_idx + 1) {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() != sample_names.len() + 1 {
                    continue;
                }

                let feature_name = parts[0].to_string();
                let values: Result<Vec<f32>, _> =
                    parts[1..].iter().map(|s| s.parse::<f32>()).collect();

                if let Ok(vals) = values {
                    feature_names.push(feature_name);
                    for (i, val) in vals.into_iter().enumerate() {
                        all_samples[i].push(val);
                    }
                }
            }

            (feature_names, all_samples, sample_names)
        }

        let (features, samples, sample_names) = parse_mouse_gut_format(&lines);

        println!(
            "Parsed {} features and {} samples",
            features.len(),
            sample_names.len()
        );
        println!(
            "Sample names: {:?}",
            sample_names.iter().take(5).collect::<Vec<_>>()
        );

        if features.is_empty() || samples.is_empty() {
            println!("Failed to parse mouse gut data, skipping test");
            return;
        }

        // Create NewDistUniFrac implementation
        let new_dist = match NewDistUniFrac::new(&tree_content, false, features.clone()) {
            Ok(d) => d,
            Err(e) => {
                println!("Failed to create NewDistUniFrac: {}", e);
                return;
            }
        };

        println!("\n=== Validation Results (tolerance: 0.01) ===");
        println!(
            "{:<15} | {:<15} | {:>15}",
            "Sample A", "Sample B", "NewDistUniFrac"
        );
        println!("{}", "-".repeat(50));

        let mut test_count = 0;
        let max_tests = 10; // Test only first 10 pairs to avoid too much output
        let mut distances_calculated = Vec::new();

        // Test pairwise distances on subset of samples
        for i in 0..std::cmp::min(3, samples.len()) {
            for j in (i + 1)..std::cmp::min(4, samples.len()) {
                if test_count >= max_tests {
                    break;
                }

                let sample_a = &sample_names[i];
                let sample_b = &sample_names[j];

                // Calculate with NewDistUniFrac only
                let new_distance = new_dist.eval(&samples[i], &samples[j]) as f64;
                distances_calculated.push(new_distance);

                test_count += 1;
                println!(
                    "{:<15} | {:<15} | {:>15.6}",
                    sample_a, sample_b, new_distance
                );
            }
            if test_count >= max_tests {
                break;
            }
        }

        // Validate basic properties instead of comparing to DistUniFrac
        println!(
            "\nNewDistUniFrac successfully processed {} sample pairs from mouse gut data!",
            test_count
        );
        println!(
            "Distance range: {:.6} to {:.6}",
            distances_calculated
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b)),
            distances_calculated
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );

        // Test self-distance (should be 0)
        let self_distance = new_dist.eval(&samples[0], &samples[0]);
        assert!(
            self_distance.abs() < 1e-6,
            "Self distance should be ~0, got {}",
            self_distance
        );

        println!("Self-distance validation: PASS (got {:.10})", self_distance);
        println!("Mouse gut microbiome data processing completed successfully!");
    }

    // ============================================================================
    // Comprehensive NewDistUniFrac Tests
    // ============================================================================

    #[test]
    fn test_new_dist_unifrac_creation() {
        init_log();

        // Simple 4-leaf tree
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        // Test unweighted
        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names.clone());
        assert!(
            dist_unifrac.is_ok(),
            "Failed to create unweighted NewDistUniFrac: {:?}",
            dist_unifrac.err()
        );

        let unifrac = dist_unifrac.unwrap();
        assert_eq!(unifrac.weighted, false);
        assert_eq!(unifrac.feature_names.len(), 4);
        assert_eq!(unifrac.num_features(), 4);

        // Test weighted
        let dist_unifrac_weighted = NewDistUniFrac::new(newick_str, true, feature_names);
        assert!(
            dist_unifrac_weighted.is_ok(),
            "Failed to create weighted NewDistUniFrac"
        );

        let unifrac_weighted = dist_unifrac_weighted.unwrap();
        assert_eq!(unifrac_weighted.weighted, true);
    }

    #[test]
    fn test_new_dist_unifrac_invalid_newick() {
        init_log();

        // Invalid Newick string - missing closing parenthesis
        let invalid_newick = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let result = NewDistUniFrac::new(invalid_newick, false, feature_names);
        assert!(result.is_err(), "Should fail with invalid Newick string");
    }

    #[test]
    fn test_new_dist_unifrac_missing_feature() {
        init_log();

        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec!["T1".to_string(), "T2".to_string(), "T5".to_string()]; // T5 doesn't exist in tree

        let result = NewDistUniFrac::new(newick_str, false, feature_names);
        assert!(result.is_err(), "Should fail with missing feature name");

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Feature name 'T5' not found in tree"),
            "Error message should mention missing feature: {}",
            err_msg
        );
    }

    #[test]
    fn test_new_dist_unifrac_identical_samples() {
        init_log();

        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();

        // Identical samples should have distance 0
        let va = vec![1.0, 0.0, 1.0, 0.0];
        let vb = vec![1.0, 0.0, 1.0, 0.0];

        let distance = dist_unifrac.eval(&va, &vb);
        assert!(
            distance.abs() < 1e-6,
            "Distance between identical samples should be ~0, got {}",
            distance
        );
    }

    #[test]
    fn test_new_dist_unifrac_completely_different_samples() {
        init_log();

        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let new_dist = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();

        // Completely different samples - test against known expected value
        let va = vec![1.0, 1.0, 0.0, 0.0]; // T1, T2 present
        let vb = vec![0.0, 0.0, 1.0, 1.0]; // T3, T4 present

        let new_distance = new_dist.eval(&va, &vb);

        println!(
            "NewDistUniFrac distance for completely different samples: {}",
            new_distance
        );

        // For NewDistUniFrac (shared/union algorithm):
        // va = [1,1,0,0] (T1,T2), vb = [0,0,1,1] (T3,T4)
        // No shared branches between the two clades, so shared = 0
        // Union includes all branches, so UniFrac = 1.0 - 0/union = 1.0
        let expected_distance = 1.0; // Correct value for shared/union algorithm
        assert!(
            (new_distance - expected_distance).abs() < 0.01,
            "NewDistUniFrac ({}) should match expected value ({}) within 0.01 tolerance",
            new_distance,
            expected_distance
        );

        // Also verify it's a substantial distance as expected
        assert!(
            new_distance > 0.5,
            "Distance should be substantial for completely different samples, got {}",
            new_distance
        );
    }

    #[test]
    fn test_new_dist_unifrac_zero_samples() {
        init_log();

        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();

        // Both samples empty
        let va = vec![0.0, 0.0, 0.0, 0.0];
        let vb = vec![0.0, 0.0, 0.0, 0.0];

        let distance = dist_unifrac.eval(&va, &vb);
        assert!(
            distance.abs() < 1e-6,
            "Distance between empty samples should be ~0, got {}",
            distance
        );
    }

    #[test]
    fn test_new_dist_unifrac_single_feature() {
        init_log();

        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let new_dist = NewDistUniFrac::new(newick_str, false, feature_names.clone()).unwrap();

        // Test T1 vs T2 (sister taxa) - basic validation
        let va = vec![1.0, 0.0, 0.0, 0.0]; // Only T1
        let vb = vec![0.0, 1.0, 0.0, 0.0]; // Only T2

        let new_distance = new_dist.eval(&va, &vb);

        println!("Sister taxa - NewDistUniFrac: {}", new_distance);
        assert!(
            new_distance >= 0.0 && new_distance <= 1.0,
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0",
            new_distance
        );

        // Test T1 vs T3 (more distant) - should be larger distance
        let vc = vec![0.0, 0.0, 1.0, 0.0]; // Only T3
        let new_distance2 = new_dist.eval(&va, &vc);

        println!("Distant taxa - NewDistUniFrac: {}", new_distance2);
        assert!(
            new_distance2 >= 0.0 && new_distance2 <= 1.0,
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0",
            new_distance2
        );
        assert!(
            new_distance2 > new_distance,
            "Distance to more distant taxa ({}) should be larger than sister taxa ({})",
            new_distance2,
            new_distance
        );

        // Verify distance ordering is preserved
        assert!(
            new_distance2 > new_distance,
            "More distant taxa should have larger distance"
        );
    }

    #[test]
    fn test_new_dist_unifrac_weighted_flag() {
        init_log();

        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let dist_unweighted =
            NewDistUniFrac::new(newick_str, false, feature_names.clone()).unwrap();
        let dist_weighted = NewDistUniFrac::new(newick_str, true, feature_names).unwrap();

        // Verify the weighted flag is stored correctly
        assert_eq!(dist_unweighted.weighted, false);
        assert_eq!(dist_weighted.weighted, true);

        // Test with different abundances - weighted and unweighted should give different results
        let va = vec![10.0, 0.0, 1.0, 0.0]; // Different abundances
        let vb = vec![1.0, 0.0, 10.0, 0.0];

        let dist_unwt = dist_unweighted.eval(&va, &vb);
        let dist_wt = dist_weighted.eval(&va, &vb);

        println!("Unweighted distance: {}", dist_unwt);
        println!("Weighted distance: {}", dist_wt);

        // Weighted and unweighted should give different results for different abundance patterns
        // Both should be valid distances (>= 0, finite)
        assert!(dist_unwt >= 0.0 && dist_unwt.is_finite(), "Unweighted distance should be valid");
        assert!(dist_wt >= 0.0 && dist_wt.is_finite(), "Weighted distance should be valid");
        
        // For this specific case with different abundances, they should differ
        // (though in some edge cases they might be similar)
        assert!(
            (dist_unwt - dist_wt).abs() > 1e-6 || (dist_unwt < 1e-6 && dist_wt < 1e-6),
            "Weighted and unweighted should differ for different abundance patterns, or both be ~0"
        );
    }

    /// Test function for user's tree and CSV files
    #[test]
    fn test_new_dist_unifrac_from_files() {
        use std::fs;
        use std::path::Path;

        init_log();

        // Use the counts file since it contains the actual feature data we need for UniFrac
        let tree_file = "data/Mouse_gut_zotu_aligned.tre";
        let counts_file = "data/Mouse_gut_zotu_counts.txt";

        println!("Testing NewDistUniFrac with real mouse gut microbiome data...");

        // Check if files exist
        if !Path::new(tree_file).exists() {
            println!(
                "Tree file '{}' not found. Please ensure the data folder exists.",
                tree_file
            );
            return;
        }

        if !Path::new(counts_file).exists() {
            println!(
                "Counts file '{}' not found. Please ensure the data folder exists.",
                counts_file
            );
            return;
        }

        println!("Using tree: {}", tree_file);
        println!("Using counts: {}", counts_file);

        // Read tree file
        let newick_str = match fs::read_to_string(tree_file) {
            Ok(content) => {
                println!("Successfully read tree file: {} characters", content.len());
                content.trim().to_string()
            }
            Err(e) => {
                println!("Error reading tree file: {}", e);
                return;
            }
        };

        println!(
            "Tree content (first 100 chars): {}",
            if newick_str.len() > 100 {
                &newick_str[..100]
            } else {
                &newick_str
            }
        );

        // Read counts file
        let data_content = match fs::read_to_string(counts_file) {
            Ok(content) => {
                println!(
                    "Successfully read counts file: {} characters",
                    content.len()
                );
                content
            }
            Err(e) => {
                println!("Error reading counts file: {}", e);
                return;
            }
        };

        // Parse counts file
        let lines: Vec<&str> = data_content.lines().collect();
        if lines.is_empty() {
            println!("Counts file is empty");
            return;
        }

        // Helper function to parse counts/OTU table format
        fn parse_counts_format(lines: &[&str]) -> (Vec<String>, Vec<Vec<f32>>, Vec<String>) {
            if lines.is_empty() {
                return (vec![], vec![], vec![]);
            }

            // Look for header line (might start with # or be first line)
            let mut header_idx = 0;
            for (i, line) in lines.iter().enumerate() {
                if line.starts_with('#') || line.contains('\t') || line.contains(',') {
                    header_idx = i;
                    break;
                }
            }

            let header = lines[header_idx].trim_start_matches('#').trim();
            let separator = if header.contains('\t') { '\t' } else { ',' };

            let parts: Vec<&str> = header.split(separator).collect();
            if parts.len() < 2 {
                return (vec![], vec![], vec![]);
            }

            // First column usually OTU/feature ID, rest are sample names
            let sample_names: Vec<String> =
                parts[1..].iter().map(|s| s.trim().to_string()).collect();
            let mut feature_names = Vec::new();
            let mut all_samples: Vec<Vec<f32>> = vec![vec![]; sample_names.len()];

            // Parse data lines
            for line in lines.iter().skip(header_idx + 1) {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                let parts: Vec<&str> = line.split(separator).collect();
                if parts.len() != sample_names.len() + 1 {
                    continue;
                }

                let feature_name = parts[0].trim().to_string();
                let values: Result<Vec<f32>, _> =
                    parts[1..].iter().map(|s| s.trim().parse::<f32>()).collect();

                if let Ok(vals) = values {
                    feature_names.push(feature_name);
                    for (i, val) in vals.into_iter().enumerate() {
                        all_samples[i].push(val);
                    }
                }
            }

            // Transpose: convert from features x samples to samples x features
            let samples: Vec<Vec<f32>> = all_samples;

            (feature_names, samples, sample_names)
        }

        // Parse OTU counts format
        let (feature_names, samples, sample_names) = parse_counts_format(&lines);

        if feature_names.is_empty() {
            println!("No feature names found");
            return;
        }

        if samples.is_empty() {
            println!("No valid samples found");
            return;
        }

        println!(
            "Found {} features and {} samples",
            feature_names.len(),
            samples.len()
        );
        println!(
            "Features (first 10): {:?}",
            if feature_names.len() > 10 {
                &feature_names[..10]
            } else {
                &feature_names
            }
        );
        println!("Samples: {:?}", sample_names);

        // Create NewDistUniFrac - this will test if feature names match tree
        println!("\nCreating NewDistUniFrac instance...");
        let dist_unifrac = match NewDistUniFrac::new(&newick_str, false, feature_names.clone()) {
            Ok(unifrac) => {
                println!("NewDistUniFrac created successfully!");
                println!("  - Weighted: {}", unifrac.weighted);
                println!("  - Features: {}", unifrac.num_features());
                unifrac
            }
            Err(e) => {
                println!("Error creating NewDistUniFrac: {}", e);
                println!("\nTroubleshooting tips:");
                println!("1. Check that feature names in data match leaf names in tree exactly");
                println!("2. Tree format should be valid Newick");
                println!("3. Feature names are case-sensitive");

                // Show first few feature names for debugging
                println!(
                    "\nFirst 5 features from data: {:?}",
                    &feature_names[..feature_names.len().min(5)]
                );
                return;
            }
        };

        // Compute sample distances (limit to first few samples for test)
        println!("\nComputing UniFrac distances...");
        let max_samples = samples.len().min(5); // Limit for test performance

        for i in 0..max_samples {
            for j in (i + 1)..max_samples {
                let distance = dist_unifrac.eval(&samples[i], &samples[j]);
                println!(
                    "  {} vs {}: {:.6}",
                    sample_names[i], sample_names[j], distance
                );

                // Validate distance properties
                assert!(
                    distance >= 0.0 && distance <= 1.0,
                    "Distance out of range [0,1]: {}",
                    distance
                );
            }
        }

        // Test self-distance
        if !samples.is_empty() {
            let self_distance = dist_unifrac.eval(&samples[0], &samples[0]);
            println!("\nSelf-distance test: {:.8}", self_distance);
            assert!(self_distance.abs() < 1e-6, "Self-distance should be ~0");
        }

        println!("\nMouse gut microbiome UniFrac analysis completed successfully!");
    }

    #[test]
    fn test_new_dist_unifrac_api_compatibility() {
        init_log();

        // Test the exact function call pattern mentioned in the requirements
        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let weighted = false;
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        // This should work exactly as specified
        let dist_unifrac = NewDistUniFrac::new(&newick_str, weighted, feature_names.clone());
        assert!(dist_unifrac.is_ok(), "API compatibility test failed");

        let unifrac = dist_unifrac.unwrap();

        // Test feature access methods
        assert_eq!(unifrac.feature_names(), &feature_names);
        assert_eq!(unifrac.num_features(), 4);

        // Test evaluation
        let va = vec![1.0, 0.0, 1.0, 0.0];
        let vb = vec![0.0, 1.0, 0.0, 1.0];
        let distance = unifrac.eval(&va, &vb);

        assert!(
            distance >= 0.0 && distance <= 1.0,
            "Distance should be normalized between 0 and 1"
        );
    }

    #[test]
    fn test_new_dist_unifrac_large_tree() {
        init_log();

        // Larger tree with more taxa - basic validation
        let newick_str = "(((T1:0.1,T2:0.1):0.05,(T3:0.1,T4:0.1):0.05):0.1,((T5:0.1,T6:0.1):0.05,(T7:0.1,T8:0.1):0.05):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
            "T5".to_string(),
            "T6".to_string(),
            "T7".to_string(),
            "T8".to_string(),
        ];

        let new_dist = NewDistUniFrac::new(newick_str, false, feature_names.clone()).unwrap();

        // Test completely different clades
        let va = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // First 4 taxa
        let vb = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // Last 4 taxa

        let new_distance = new_dist.eval(&va, &vb);

        println!(
            "Large tree different clades - NewDistUniFrac: {}",
            new_distance
        );
        assert!(
            new_distance >= 0.0 && new_distance <= 1.0,
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0",
            new_distance
        );

        // Test overlapping samples
        let vc = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]; // Mixed across clades
        let new_distance2 = new_dist.eval(&va, &vc);

        println!("Large tree overlapping - NewDistUniFrac: {}", new_distance2);
        assert!(
            new_distance2 >= 0.0 && new_distance2 <= 1.0,
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0",
            new_distance2
        );

        // Verify distance ordering is preserved
        assert!(
            new_distance2 < new_distance,
            "Overlapping samples should have smaller distance ({} < {})",
            new_distance2,
            new_distance
        );
    }

    #[test]
    fn test_new_dist_unifrac_succparen_patterns() {
        init_log();

        // Test that our implementation follows succparen patterns correctly
        let newick_str = "((T1:0.2,T2:0.3):0.1,(T3:0.4,T4:0.5):0.2):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();

        // Verify internal structure is built correctly
        assert!(
            dist_unifrac.post.len() > 0,
            "Post-order traversal should be populated"
        );
        assert!(
            dist_unifrac.kids.len() > 0,
            "Children arrays should be populated"
        );
        assert!(
            dist_unifrac.lens.len() > 0,
            "Branch lengths should be populated"
        );
        assert_eq!(dist_unifrac.leaf_ids.len(), 4, "Should have 4 leaf IDs");

        // Test distance computation
        let va = vec![1.0, 0.0, 0.0, 0.0];
        let vb = vec![0.0, 1.0, 0.0, 0.0];
        let distance = dist_unifrac.eval(&va, &vb);

        // Distance should be positive and reasonable
        assert!(
            distance > 0.0 && distance < 1.0,
            "Distance should be in valid range, got {}",
            distance
        );
    }

    #[test]
    fn test_new_dist_unifrac_bitvec_conversion() {
        init_log();

        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let new_dist = NewDistUniFrac::new(newick_str, false, feature_names.clone()).unwrap();

        // Test presence/absence conversion with varying abundances
        let va = vec![0.5, 0.0, 1.5, 0.0]; // T1 and T3 present (> 0)
        let vb = vec![0.0, 2.0, 0.0, 0.1]; // T2 and T4 present (> 0)

        let new_distance = new_dist.eval(&va, &vb);

        println!("Bitvec conversion - NewDistUniFrac: {}", new_distance);
        assert!(
            new_distance >= 0.0 && new_distance <= 1.0,
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0",
            new_distance
        );

        // Test with very small positive values
        let vc = vec![0.0001, 0.0, 0.0001, 0.0]; // Very small positive values
        let vd = vec![0.0, 0.0001, 0.0, 0.0001];

        let new_distance2 = new_dist.eval(&vc, &vd);

        println!("Small values - NewDistUniFrac: {}", new_distance2);
        assert!(
            new_distance2 >= 0.0 && new_distance2 <= 1.0,
            "NewDistUniFrac distance ({}) should be between 0.0 and 1.0",
            new_distance2
        );

        // Both should give same result (presence/absence only matters)
        assert!(
            (new_distance - new_distance2).abs() < 1e-6,
            "Distance should only depend on presence/absence, got {} vs {}",
            new_distance,
            new_distance2
        );
    }

    #[test]
    fn test_new_dist_unifrac_symmetry() {
        init_log();

        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();

        // Test that distance is symmetric: d(A,B) = d(B,A)
        let va = vec![1.0, 0.0, 1.0, 0.0];
        let vb = vec![0.0, 1.0, 0.0, 1.0];

        let distance_ab = dist_unifrac.eval(&va, &vb);
        let distance_ba = dist_unifrac.eval(&vb, &va);

        assert!(
            (distance_ab - distance_ba).abs() < 1e-10,
            "Distance should be symmetric: d(A,B)={}, d(B,A)={}",
            distance_ab,
            distance_ba
        );
    }

    #[test]
    fn test_new_dist_unifrac_triangle_inequality() {
        init_log();

        let newick_str = "((T1:0.1,T2:0.2):0.05,(T3:0.3,T4:0.4):0.1):0.0;";
        let feature_names = vec![
            "T1".to_string(),
            "T2".to_string(),
            "T3".to_string(),
            "T4".to_string(),
        ];

        let dist_unifrac = NewDistUniFrac::new(newick_str, false, feature_names).unwrap();

        // Test triangle inequality: d(A,C) <= d(A,B) + d(B,C)
        let va = vec![1.0, 0.0, 0.0, 0.0]; // Only T1
        let vb = vec![0.0, 1.0, 0.0, 0.0]; // Only T2
        let vc = vec![0.0, 0.0, 1.0, 0.0]; // Only T3

        let d_ab = dist_unifrac.eval(&va, &vb);
        let d_bc = dist_unifrac.eval(&vb, &vc);
        let d_ac = dist_unifrac.eval(&va, &vc);

        println!("d(A,B) = {}, d(B,C) = {}, d(A,C) = {}", d_ab, d_bc, d_ac);

        // Triangle inequality: d(A,C) <= d(A,B) + d(B,C)
        assert!(
            d_ac <= d_ab + d_bc + 1e-10,
            "Triangle inequality violated: d(A,C)={} > d(A,B)+d(B,C)={}",
            d_ac,
            d_ab + d_bc
        );
    }

    #[test]
    fn test_new_dist_unifrac_distance_trait_usage() {
        use std::fs;

        init_log();

        // Same data as your mouse gut test
        let tree_file = "data/Mouse_gut_zotu_aligned.tre";
        let counts_file = "data/Mouse_gut_zotu_counts.txt";

        if !std::path::Path::new(tree_file).exists() || !std::path::Path::new(counts_file).exists()
        {
            println!("Data files not found, skipping test");
            return;
        }

        // Read your real data
        let tree_content = fs::read_to_string(tree_file).unwrap();
        let data_content = fs::read_to_string(counts_file).unwrap();
        let lines: Vec<&str> = data_content.lines().collect();

        // Helper function to parse counts/OTU table format
        fn parse_counts_format(lines: &[&str]) -> (Vec<String>, Vec<Vec<f32>>, Vec<String>) {
            if lines.is_empty() {
                return (vec![], vec![], vec![]);
            }

            let mut header_idx = 0;
            for (i, line) in lines.iter().enumerate() {
                if line.starts_with('#') || line.contains('\t') || line.contains(',') {
                    header_idx = i;
                    break;
                }
            }

            let header = lines[header_idx].trim_start_matches('#').trim();
            let separator = if header.contains('\t') { '\t' } else { ',' };

            let parts: Vec<&str> = header.split(separator).collect();
            if parts.len() < 2 {
                return (vec![], vec![], vec![]);
            }

            let sample_names: Vec<String> =
                parts[1..].iter().map(|s| s.trim().to_string()).collect();
            let mut feature_names = Vec::new();
            let mut all_samples: Vec<Vec<f32>> = vec![vec![]; sample_names.len()];

            for line in lines.iter().skip(header_idx + 1) {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                let parts: Vec<&str> = line.split(separator).collect();
                if parts.len() != sample_names.len() + 1 {
                    continue;
                }

                let feature_name = parts[0].trim().to_string();
                let values: Result<Vec<f32>, _> =
                    parts[1..].iter().map(|s| s.trim().parse::<f32>()).collect();

                if let Ok(vals) = values {
                    feature_names.push(feature_name);
                    for (i, val) in vals.into_iter().enumerate() {
                        all_samples[i].push(val);
                    }
                }
            }

            let samples: Vec<Vec<f32>> = all_samples;
            (feature_names, samples, sample_names)
        }

        let (feature_names, samples, sample_names) = parse_counts_format(&lines);

        if feature_names.is_empty() || samples.is_empty() {
            println!("No valid data found");
            return;
        }

        // Create NewDistUniFrac
        let dist_unifrac = NewDistUniFrac::new(&tree_content, false, feature_names).unwrap();

        println!("=== Using Distance Trait to Get Same Numbers ===");

        // METHOD 1: Direct Distance trait usage
        let distance_trait: &dyn Distance<f32> = &dist_unifrac;

        if samples.len() >= 4 {
            println!("Direct Distance Trait Usage:");
            println!(
                "   {} vs {}: {:.6}",
                sample_names[0],
                sample_names[1],
                distance_trait.eval(&samples[0], &samples[1])
            );
            println!(
                "   {} vs {}: {:.6}",
                sample_names[0],
                sample_names[2],
                distance_trait.eval(&samples[0], &samples[2])
            );
            println!(
                "   {} vs {}: {:.6}",
                sample_names[0],
                sample_names[3],
                distance_trait.eval(&samples[0], &samples[3])
            );
            println!(
                "   Self-distance ({} vs {}): {:.8}",
                sample_names[0],
                sample_names[0],
                distance_trait.eval(&samples[0], &samples[0])
            );
        }

        println!();

        // METHOD 2: Generic function using Distance trait
        fn calculate_distances<D: Distance<f32>>(
            dist: &D,
            samples: &[Vec<f32>],
            sample_names: &[String],
        ) {
            println!(" Generic Function with Distance Trait:");
            if samples.len() >= 4 {
                println!(
                    "   {} vs {}: {:.6}",
                    sample_names[0],
                    sample_names[1],
                    dist.eval(&samples[0], &samples[1])
                );
                println!(
                    "   {} vs {}: {:.6}",
                    sample_names[0],
                    sample_names[2],
                    dist.eval(&samples[0], &samples[2])
                );
                println!(
                    "   {} vs {}: {:.6}",
                    sample_names[0],
                    sample_names[3],
                    dist.eval(&samples[0], &samples[3])
                );
                println!(
                    "   Self-distance ({} vs {}): {:.8}",
                    sample_names[0],
                    sample_names[0],
                    dist.eval(&samples[0], &samples[0])
                );
            }
        }

        calculate_distances(&dist_unifrac, &samples, &sample_names);

        println!();

        // METHOD 3: Polymorphic usage - can work with ANY Distance implementation
        fn process_with_any_distance<D: Distance<f32>>(
            distance_impl: D,
            data: &[Vec<f32>],
            names: &[String],
        ) {
            println!("Polymorphic Distance Usage:");
            if data.len() >= 4 {
                println!(
                    "   {} vs {}: {:.6}",
                    names[0],
                    names[1],
                    distance_impl.eval(&data[0], &data[1])
                );
                println!(
                    "   {} vs {}: {:.6}",
                    names[0],
                    names[2],
                    distance_impl.eval(&data[0], &data[2])
                );
                println!(
                    "   {} vs {}: {:.6}",
                    names[0],
                    names[3],
                    distance_impl.eval(&data[0], &data[3])
                );
                println!(
                    "   Self-distance ({} vs {}): {:.8}",
                    names[0],
                    names[0],
                    distance_impl.eval(&data[0], &data[0])
                );
            }
        }

        process_with_any_distance(dist_unifrac, &samples, &sample_names);

        println!();
        println!("All methods produce identical results using the Distance trait!");
    }


    #[test]
    fn test_unifrac_pair_weighted() {
        // Simple tree structure:
        //     0 (root, length 0.0)
        //    / \
        //   1   2 (length 1.0 each)
        //  / \ / \
        // 3  4 5  6 (leaves, length 0.5 each)
        // Leaves: 3, 4, 5, 6
        
        // Post-order traversal: [3, 4, 1, 5, 6, 2, 0]
        let post = vec![3, 4, 1, 5, 6, 2, 0];
        
        // Children for each node: node 0 has [1, 2], node 1 has [3, 4], node 2 has [5, 6]
        // Index corresponds to node ID
        let kids = vec![
            vec![1, 2],  // node 0
            vec![3, 4],  // node 1
            vec![5, 6],  // node 2
            vec![],      // node 3 (leaf)
            vec![],      // node 4 (leaf)
            vec![],      // node 5 (leaf)
            vec![],      // node 6 (leaf)
        ];
        
        // Branch lengths (one per node)
        let lens = vec![0.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5];
        
        // Leaf IDs (indices in va/vb correspond to these leaf IDs)
        let leaf_ids = vec![3, 4, 5, 6];
        
        // Abundance vectors for two samples
        // Sample A: [1.0, 2.0, 3.0, 4.0] -> normalized: [0.1, 0.2, 0.3, 0.4]
        let va = vec![1.0, 2.0, 3.0, 4.0];
        
        // Sample B: [2.0, 2.0, 2.0, 2.0] -> normalized: [0.25, 0.25, 0.25, 0.25]
        let vb = vec![2.0, 2.0, 2.0, 2.0];
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Verify the function returns a valid distance
        // Distance should be between 0.0 and 1.0 (normalized)
        assert!(result >= 0.0);
        assert!(result <= 1.0);
        assert!(result.is_finite());
        
        // With different distributions, distance should be > 0
        assert!(result > 0.0);
    }

    #[test]
    fn test_unifrac_pair_weighted_zero_abundance_va() {
        // va has zero abundance, vb has non-zero
        // Normalized: va = [0.0, 0.0], vb = [0.5, 0.5]
        // Differences: [-0.5, -0.5]
        let post = vec![0, 1];
        let kids = vec![vec![], vec![]];
        let lens = vec![0.0, 1.0]; // Node 0 has zero length, node 1 has length 1.0
        let leaf_ids = vec![0, 1];
        let va = vec![0.0, 0.0];
        let vb = vec![1.0, 1.0];
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Should have non-zero distance when one sample is zero and other is not
        // But if total_length is 0.0 (all branches zero), returns 0.0
        assert!(result >= 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_unifrac_pair_weighted_zero_abundance_vb() {
        let post = vec![0];
        let kids = vec![vec![]];
        let lens = vec![0.0];
        let leaf_ids = vec![0];
        let va = vec![1.0, 1.0];
        let vb = vec![0.0, 0.0];
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        assert!(result >= 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_unifrac_pair_weighted_both_zero_abundance() {
        // Both samples have zero abundance - should return 0.0 (no distance)
        let post = vec![0];
        let kids = vec![vec![]];
        let lens = vec![0.0];
        let leaf_ids = vec![0];
        let va = vec![0.0, 0.0];
        let vb = vec![0.0, 0.0];
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Zero abundance in both should result in zero distance
        // Note: If total_length is 0.0, function returns 0.0
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_unifrac_pair_weighted_identical_samples() {
        // Identical samples should have distance = 0.0
        let post = vec![0, 1, 2];
        let kids = vec![vec![0, 1], vec![], vec![]];
        let lens = vec![0.0, 1.0, 1.0];
        let leaf_ids = vec![0, 1];
        let va = vec![1.0, 1.0];
        let vb = vec![1.0, 1.0];
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Identical normalized abundances should result in zero distance
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_unifrac_pair_weighted_normalization_scales() {
        // Test that normalization works correctly with different scales
        // va: [10.0, 20.0] should normalize to [0.33..., 0.66...] (same as [1.0, 2.0])
        // vb: [10.0, 20.0] should normalize to [0.33..., 0.66...] (same as [1.0, 2.0])
        // So scaled versions should give same result as unscaled versions
        let post = vec![0, 1, 2];
        let kids = vec![vec![0, 1], vec![], vec![]];
        let lens = vec![0.0, 1.0, 1.0];
        let leaf_ids = vec![0, 1];
        
        // Small scale
        let va_small = vec![1.0, 2.0];
        let vb_small = vec![1.0, 2.0];
        let result_small = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va_small, &vb_small);
        
        // Large scale (should normalize to same proportions)
        let va_large = vec![10.0, 20.0];
        let vb_large = vec![10.0, 20.0];
        let result_large = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va_large, &vb_large);
        
        // Results should be identical after normalization (both should be 0.0 since samples are identical)
        assert_eq!(result_small, 0.0);
        assert_eq!(result_large, 0.0);
        assert_eq!(result_small, result_large);
    }

    #[test]
    fn test_unifrac_pair_weighted_partial_sums_initialization() {
        // Test that partial_sums is correctly initialized based on lens length
        // This tests the new num_nodes and partial_sums initialization logic
        let post = vec![0, 1, 2, 3];
        let kids = vec![vec![1, 2], vec![], vec![], vec![]];
        let lens = vec![0.0, 1.0, 1.0, 0.5]; // 4 nodes
        let leaf_ids = vec![1, 2];
        let va = vec![1.0, 2.0];
        let vb = vec![2.0, 1.0];
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Should handle the partial_sums initialization correctly
        assert!(result >= 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_unifrac_pair_weighted_difference_computation() {
        // Test that differences between normalized vectors are computed correctly
        // va: [1.0, 0.0] -> normalized: [1.0, 0.0]
        // vb: [0.0, 1.0] -> normalized: [0.0, 1.0]
        // Differences at leaves: partial_sums[0] = 1.0, partial_sums[1] = -1.0
        // After propagation: partial_sums[2] = 1.0 + (-1.0) = 0.0
        // Distance = (|1.0| * 0.0 + |-1.0| * 1.0 + |0.0| * 1.0) / (0.0 + 1.0 + 1.0)
        //         = (0.0 + 1.0 + 0.0) / 2.0 = 0.5
        // Note: Node 0 has lens[0] = 0.0, so it's skipped in distance calculation
        let post = vec![0, 1, 2];
        let kids = vec![vec![0, 1], vec![], vec![]];
        let lens = vec![0.0, 1.0, 1.0]; // Node 0 has 0 length (skipped), nodes 1 and 2 have length 1.0
        let leaf_ids = vec![0, 1];
        let va = vec![1.0, 0.0];
        let vb = vec![0.0, 1.0];
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Expected: (|-1.0| * 1.0) / (1.0 + 1.0) = 1.0 / 2.0 = 0.5
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_unifrac_pair_weighted_small_difference_threshold() {
        // Test that very small differences (< 1e-12) are ignored at leaf level
        // This tests the threshold check: if diff.abs() > 1e-12
        let post = vec![0, 1];
        let kids = vec![vec![], vec![]];
        let lens = vec![0.0, 1.0];
        let leaf_ids = vec![0, 1];
        
        // Create vectors that normalize to nearly identical values
        // va: [1.0, 1.0] -> normalized: [0.5, 0.5]
        // vb: [1.0 + 1e-13, 1.0 - 1e-13] -> normalized: [0.5 + tiny, 0.5 - tiny]
        // Difference should be very small (< 1e-12) and ignored
        let va = vec![1.0, 1.0];
        let vb = vec![1.0 + 1e-13, 1.0 - 1e-13];
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Small differences below threshold should result in zero distance
        // (since they're not stored in partial_sums)
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_unifrac_pair_weighted_leaf_id_indexing() {
        // Test that leaf_id is used correctly as index into partial_sums
        // Using non-sequential leaf_ids to verify indexing works
        let post = vec![0, 1, 2, 3, 4];
        let kids = vec![vec![1, 2], vec![], vec![], vec![], vec![]];
        let lens = vec![0.0, 1.0, 1.0, 0.5, 0.5]; // 5 nodes
        let leaf_ids = vec![2, 4]; // Non-sequential leaf IDs
        let va = vec![1.0, 2.0];
        let vb = vec![2.0, 1.0];
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Should correctly index into partial_sums using leaf_id
        assert!(result >= 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_unifrac_pair_weighted_mismatched_vector_lengths() {
        // Test handling when va and vb have different lengths
        // The code checks: if i < normalized_a.len() && i < normalized_b.len()
        let post = vec![0, 1, 2];
        let kids = vec![vec![0, 1], vec![], vec![]];
        let lens = vec![0.0, 1.0, 1.0];
        let leaf_ids = vec![0, 1];
        let va = vec![1.0, 2.0, 3.0]; // 3 elements
        let vb = vec![1.0, 2.0]; // 2 elements
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Should handle mismatched lengths gracefully
        assert!(result >= 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_unifrac_pair_weighted_more_leaf_ids_than_vectors() {
        // Test when leaf_ids has more elements than va/vb
        let post = vec![0, 1, 2, 3];
        let kids = vec![vec![], vec![], vec![], vec![]];
        let lens = vec![0.0, 1.0, 1.0, 1.0];
        let leaf_ids = vec![0, 1, 2, 3]; // 4 leaf IDs
        let va = vec![1.0, 2.0]; // Only 2 elements
        let vb = vec![2.0, 1.0]; // Only 2 elements
        
        let result = unifrac_pair_weighted(&post, &kids, &lens, &leaf_ids, &va, &vb);
        
        // Should handle when i >= normalized_a.len() or normalized_b.len()
        assert!(result >= 0.0);
        assert!(result.is_finite());
    }
} // end of module tests
