# Workspace Map for anndists

This document provides a comprehensive overview of the anndists workspace structure for future coding agents.

## Project Overview

**anndists** is a Rust crate that provides distance computation implementations used in related crates:
- [hnsw_rs](https://crates.io/crates/hnsw_rs)
- [annembed](https://crates.io/crates/annembed)
- [coreset](https://github.com/jean-pierreBoth/coreset)

The crate implements various distance metrics with SIMD optimizations for performance-critical use cases.

## Directory Structure

```
anndists/
  src/                    # Main source code directory
    lib.rs             # Library entry point, logger initialization
    prelude.rs         # Public API exports
    dist/              # Distance implementations module
      mod.rs         # Module declarations and re-exports
      traits.rs      # Distance trait definition
      utils.rs       # Utility functions (l2_normalize)
      basic.rs       # Basic distances (L1, L2, Cosine, Dot)
      probability.rs # Probability distances (Hellinger, Jeffreys, Jensen-Shannon)
      set.rs         # Set distances (Hamming, Jaccard)
      string.rs      # String distances (Levenshtein)
      custom.rs      # Custom distance wrappers (NoDist, DistFn, DistPtr, DistCFFI)
      unifrac.rs     # UniFrac phylogenetic distances
      distances.rs   # Test module (test-only, ~2,045 lines)
      disteez.rs     # SIMD implementations using simdeez crate (x86_64)
      distsimd.rs    # SIMD implementations using std::simd (nightly)
  examples/               # Example code
    test_unifrac_files.rs
  data/                   # Test data files
    Mouse_gut_embedded.csv
    Mouse_gut_zotu_aligned.tre
    Mouse_gut_zotu_counts.txt
    test_OTU_table.txt
    test_truth.txt
    test.nwk
  ExternC_UniFrac/        # External C library for UniFrac
    libunifrac.so
  Cargo.toml              # Rust project configuration
  build.rs                # Build script
  README.md               # Project documentation
```

## Core Architecture

### Module Structure

#### `src/lib.rs`
- Library entry point
- Initializes logging facility using `env_logger`
- Declares `dist` and `prelude` modules
- Uses `lazy_static` for logger initialization

#### `src/prelude.rs`
- Public API exports
- Re-exports the `Distance` trait for easy access

#### `src/dist/mod.rs`
- Module organization for distance implementations
- Declares and re-exports all distance modules:
  - `traits` - Core Distance trait
  - `utils` - Utility functions
  - `basic` - Basic vector distances
  - `probability` - Probability distribution distances
  - `set` - Set-based distances
  - `string` - String distances
  - `custom` - Custom distance wrappers
  - `unifrac` - UniFrac phylogenetic distances
  - `distances` - Test module (test-only)
- Declares SIMD modules: `distsimd`, `disteez`

#### `src/dist/traits.rs` (~35 lines)
**Core trait definition**

- `Distance<T: Send + Sync>`: Trait defining distance computation interface
  - Method: `fn eval(&self, va: &[T], vb: &[T]) -> f32`

#### `src/dist/utils.rs` (~28 lines)
**Utility functions**

- `l2_normalize(va: &mut [f32])` - Normalizes vector to unit L2 norm

#### `src/dist/basic.rs` (~243 lines)
**Basic vector distance metrics**

1. **DistL1** - L1 (Manhattan) distance
   - Types: i32, f64, i64, u32, u16, u8, f32 (with SIMD)
   
2. **DistL2** - L2 (Euclidean) distance
   - Types: i32, f64, i64, u32, u16, u8, f32 (with SIMD)
   
3. **DistCosine** - Cosine distance
   - Types: f32, f64, i64, i32, u16
   
4. **DistDot** - Dot product distance (for pre-normalized vectors)
   - Type: f32 (with SIMD optimizations)

#### `src/dist/probability.rs` (~160 lines)
**Probability distribution distances**

5. **DistHellinger** - Hellinger distance
   - Types: f32, f64
   
6. **DistJeffreys** - Jeffreys divergence (symmetrized KL divergence)
   - Types: f32, f64
   - Note: Does not satisfy triangle inequality
   
7. **DistJensenShannon** - Jensen-Shannon distance
   - Types: f32, f64
   - Bounded metric (square root of JS divergence)

#### `src/dist/set.rs` (~158 lines)
**Set-based distance metrics**

8. **DistHamming** - Hamming distance
   - Types: i32, i64, f32, f64, u32, u64, u16, u8
   
9. **DistJaccard** - Jaccard distance
   - Types: u8, u16, u32

#### `src/dist/string.rs` (~57 lines)
**String distance metrics**

10. **DistLevenshtein** - Levenshtein edit distance
    - Type: u16 (strings)

#### `src/dist/custom.rs` (~108 lines)
**Custom distance wrappers**

11. **NoDist** - Placeholder distance (panics on eval)
    - Used when reloading graph data without distance computation
    
12. **DistFn** - Closure-based distance
    - Allows custom Rust closures as distances
    
13. **DistPtr** - Function pointer distance
    - Allows Rust function pointers as distances
    
14. **DistCFFI** - Generic C function pointer distance
    - Allows custom C distance functions
    
15. **DistCFnPtr** - Type alias for C function pointer

#### `src/dist/unifrac.rs` (~1,056 lines)
**UniFrac phylogenetic distance implementations**

16. **DistUniFrac** - Original UniFrac implementation
    - Uses phylotree crate
    - Requires Newick tree format
    - Supports weighted and unweighted variants
    
17. **NewDistUniFrac** - High-performance UniFrac implementation
    - Uses succparen crate for balanced parentheses tree representation
    - Optimized for large-scale computations
    - Bit-vector based implementation
    
18. **DistUniFracCFFI** - UniFrac via C FFI (commented out)
    - Links to external C library (libunifrac.so)
    
19. **DistUniFrac_C** - C FFI context struct
    - Holds parameters for C++ UniFrac library
    
20. **OpaqueBPTree** - Opaque pointer type for C++ tree structure
    
21. **ComputeStatus** - Status enum for C FFI operations

**Helper Functions:**
- `build_tint_lint()` - Builds parent/edge arrays from tree
- `build_leaf_map()` - Maps leaf names to postorder indices
- `extract_leaf_names_from_newick_string()` - Extracts leaf names
- `unifrac_pair()` - Fast unweighted UniFrac using bit masks
- `compute_unifrac_for_pair_unweighted_bitwise()` - Unweighted UniFrac
- `compute_unifrac_for_pair_weighted_bitwise()` - Weighted UniFrac
- `make_presence_mask_f32()` - Creates presence masks (with AVX2 optimization)
- `or_masks()`, `extract_set_bits()` - Bit manipulation utilities

#### `src/dist/distances.rs` (~2,045 lines)
**Test module** (test-only, compiled only in test mode)

- Contains comprehensive tests for all distance implementations
- Tests are organized by distance type
- Includes integration tests for UniFrac with real data files

#### `src/dist/disteez.rs` (~375 lines)
**SIMD implementations using simdeez crate** (x86_64 only)

- Requires feature: `simdeez_f`
- Provides AVX2 and SSE2 implementations
- Functions:
  - `distance_l1_f32_avx2`
  - `distance_l2_f32_avx2`
  - `distance_dot_f32_avx2` / `distance_dot_f32_sse2`
  - `distance_hellinger_f32_avx2`
  - `distance_jeffreys_f32_avx2`
  - `distance_hamming_i32_avx2`
  - `distance_hamming_f64_avx2`

#### `src/dist/distsimd.rs` (~350 lines)
**SIMD implementations using std::simd** (nightly Rust)

- Requires feature: `stdsimd` and nightly compiler
- Uses portable_simd feature
- Functions:
  - `distance_l1_f32_simd`
  - `distance_l2_f32_simd`
  - `distance_dot_f32_simd_iter`
  - `distance_jaccard_u32_16_simd`
  - `distance_jaccard_f32_16_simd`
  - `distance_jaccard_u64_8_simd`

## Features

### Cargo Features

1. **default** - No features enabled by default
2. **simdeez_f** - Enables SIMD via simdeez crate (x86_64)
3. **stdsimd** - Enables std::simd (requires nightly Rust)

### Build Configuration

- **Edition**: 2021
- **Release Profile**: LTO enabled, opt-level 3
- **Target**: Library crate (rlib)

## Dependencies

### Core Dependencies
- `cfg-if` - Conditional compilation
- `rayon` - Parallel processing
- `num_cpus` - CPU detection
- `cpu-time` - CPU time measurement
- `num-traits` - Numeric trait definitions
- `rand` - Random number generation
- `lazy_static` - Lazy static initialization
- `log` / `env_logger` - Logging
- `anyhow` - Error handling
- `phylotree` - Phylogenetic tree handling (git dependency)
- `bitvec` - Bit vector operations
- `succparen` - Balanced parentheses tree operations
- `newick` - Newick tree format parsing

### Optional Dependencies
- `simdeez` - SIMD abstractions (feature: simdeez_f)

### Dev Dependencies
- `rand` (with std_rng feature) - For tests

## Key Design Patterns

1. **Zero-sized structs** - Distance types are unit structs (no data, just behavior)
2. **Trait-based polymorphism** - All distances implement `Distance<T>` trait
3. **Macro-based implementations** - Uses macros for repetitive implementations across types
4. **Conditional compilation** - SIMD features gated behind features and architecture checks
5. **Runtime feature detection** - Uses `is_x86_feature_detected!` for AVX2/SSE2
6. **Modular organization** - Code split into focused modules by distance category

## Testing

- Tests are embedded in `src/dist/distances.rs` (test-only module)
- Uses `rand` for generating test vectors
- Validates SIMD implementations against scalar implementations
- Test data files in `data/` directory
- Run tests: `cargo test --lib`

## Build Instructions

### Standard Build
```bash
cargo build --release
```

### With SIMD (x86_64)
```bash
cargo build --release --features "simdeez_f"
```

### With std::simd (nightly)
```bash
cargo build --release --features "stdsimd"
```

## Common Tasks

### Adding a New Distance

1. Choose appropriate module (basic, probability, set, string, custom, or create new)
2. Define a zero-sized struct: `pub struct DistNew;`
3. Implement `Distance<T>` trait for desired types
4. Optionally add SIMD implementations in `disteez.rs` or `distsimd.rs`
5. Add tests to `distances.rs` test module

### Modifying SIMD Code

- **x86_64 SIMD**: Edit `src/dist/disteez.rs`
- **Portable SIMD**: Edit `src/dist/distsimd.rs`
- Ensure feature gates are correct (`#[cfg(feature = "...")]`)

### Debugging

- Enable logging: Set `RUST_LOG=debug` environment variable
- Verbose feature: `cargo build --features verbose_1` (if enabled)

## Module Organization

The codebase has been refactored from a monolithic `distances.rs` file into focused modules:

- **traits.rs** - Core trait definition
- **utils.rs** - Shared utility functions
- **basic.rs** - Basic vector distances (L1, L2, Cosine, Dot)
- **probability.rs** - Probability distribution distances
- **set.rs** - Set-based distances (Hamming, Jaccard)
- **string.rs** - String distances (Levenshtein)
- **custom.rs** - Custom distance wrappers
- **unifrac.rs** - UniFrac phylogenetic distances (largest module)
- **distances.rs** - Test module (test-only)

All modules are re-exported through `mod.rs` for backward compatibility.

## Notes for Future Agents

1. **Modular Structure**: Code is now organized into focused modules - add new distances to appropriate module
2. **SIMD Safety**: All SIMD functions are `unsafe` - ensure proper bounds checking
3. **Feature Gates**: Always check feature gates when modifying SIMD code
4. **Test Coverage**: Run tests for both scalar and SIMD paths when modifying distance implementations
5. **Architecture**: x86_64 specific code is behind `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]`
6. **UniFrac Complexity**: UniFrac implementations are complex - understand tree structures before modifying
7. **Import Paths**: Use `anndists::dist::*` to access distance types (not `anndists::dist::distances::*`)

## External Resources

- [simdeez crate](https://crates.io/crates/simdeez)
- [phylotree-rs](https://github.com/lucblassel/phylotree-rs)
- [succparen crate](https://crates.io/crates/succparen)
- Related crates: hnsw_rs, annembed, coreset
