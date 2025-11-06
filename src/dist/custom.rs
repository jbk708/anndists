//! Custom distance implementations: NoDist, DistFn, DistPtr, and DistCFFI.

use super::traits::Distance;
use num_traits::float::Float;
use std::os::raw::c_ulonglong;

/// Special forbidden computation distance. It is associated to a unit NoData structure
/// This is a special structure used when we want to only reload the graph from a previous computation
/// possibly from an foreign language (and we do not have access to the original type of data from the foreign language).
#[derive(Default, Copy, Clone)]
pub struct NoDist;

impl<T: Send + Sync> Distance<T> for NoDist {
    fn eval(&self, _va: &[T], _vb: &[T]) -> f32 {
        log::error!("panic error : cannot call eval on NoDist");
        panic!("cannot call distance with NoDist");
    }
} // end impl block for NoDist

//=======================================================================================
//   Case of function pointers (cover Trait Fn , FnOnce ...)
// The book (Function item types):  " There is a coercion from function items to function pointers with the same signature  "
// The book (Call trait and coercions): "Non capturing closures can be coerced to function pointers with the same signature"

/// This type is for function with a C-API
/// Distances can be computed by such a function. It
/// takes as arguments the two (C, rust, julia) pointers to primitive type vectos and length
/// passed as a unsignedlonlong (64 bits) which is called c_ulonglong in Rust and Culonglong in Julia
///
pub type DistCFnPtr<T> = extern "C" fn(*const T, *const T, len: c_ulonglong) -> f32;

/// A structure to implement Distance Api for type DistCFnPtr\<T\>,
/// i.e distance provided by a C function pointer.  
/// It must be noted that this can be used in Julia via the macro @cfunction
/// to define interactiveley a distance function , compile it on the fly and sent it
/// to Rust via the init_hnsw_{f32, i32, u16, u32, u8} function
/// defined in libext
///
pub struct DistCFFI<T: Copy + Clone + Sized + Send + Sync> {
    dist_function: DistCFnPtr<T>,
}

impl<T: Copy + Clone + Sized + Send + Sync> DistCFFI<T> {
    pub fn new(f: DistCFnPtr<T>) -> Self {
        DistCFFI { dist_function: f }
    }
}

impl<T: Copy + Clone + Sized + Send + Sync> Distance<T> for DistCFFI<T> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        // get pointers
        let len = va.len();
        let ptr_a = va.as_ptr();
        let ptr_b = vb.as_ptr();
        let dist = (self.dist_function)(ptr_a, ptr_b, len as c_ulonglong);
        log::trace!(
            "DistCFFI dist_function_ptr {:?} returning {:?} ",
            self.dist_function,
            dist
        );
        dist
    } // end of compute
} // end of impl block

//========================================================================================================

/// This structure is to let user define their own distance with closures.
pub struct DistFn<T: Copy + Clone + Sized + Send + Sync> {
    dist_function: Box<dyn Fn(&[T], &[T]) -> f32 + Send + Sync>,
}

impl<T: Copy + Clone + Sized + Send + Sync> DistFn<T> {
    /// construction of a DistFn
    pub fn new(f: Box<dyn Fn(&[T], &[T]) -> f32 + Send + Sync>) -> Self {
        DistFn { dist_function: f }
    }
}

impl<T: Copy + Clone + Sized + Send + Sync> Distance<T> for DistFn<T> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        (self.dist_function)(va, vb)
    }
}

//=======================================================================================

/// This structure uses a Rust function pointer to define the distance.
/// For commodity it can build upon a fonction returning a f64.
/// Beware that if F is f64, the distance converted to f32 can overflow!

#[derive(Copy, Clone)]
pub struct DistPtr<T: Copy + Clone + Sized + Send + Sync, F: Float> {
    dist_function: fn(&[T], &[T]) -> F,
}

impl<T: Copy + Clone + Sized + Send + Sync, F: Float> DistPtr<T, F> {
    /// construction of a DistPtr
    pub fn new(f: fn(&[T], &[T]) -> F) -> Self {
        DistPtr { dist_function: f }
    }
}

/// beware that if F is f64, the distance converted to f32 can overflow!
impl<T: Copy + Clone + Sized + Send + Sync, F: Float> Distance<T> for DistPtr<T, F> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        (self.dist_function)(va, vb).to_f32().unwrap()
    }
}
