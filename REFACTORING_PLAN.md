# Refactoring Plan: Breaking up `distances.rs`

## Current State
- **Original File**: `src/dist/distances.rs`
- **Original Size**: ~3,935 lines total
  - Implementation code: ~1,910 lines
  - Test code: ~2,025 lines (lines 1911-3935)
- **Current Structure**: Mostly refactored - 7 modules extracted, only UniFrac remains
- **Remaining in distances.rs**: ~2,000+ lines (UniFrac implementations + tests)

## Progress Status

### ✅ Completed Modules
1. **`src/dist/traits.rs`** ✅ (renamed from `trait.rs` - `trait` is a Rust keyword)
   - Status: Complete and tested
   - Contains: `Distance<T>` trait definition
   - Commits: `354b9b5`

2. **`src/dist/utils.rs`** ✅
   - Status: Complete and tested
   - Contains: `l2_normalize()` function
   - Commits: `798ae91`

3. **`src/dist/basic.rs`** ✅
   - Status: Complete and tested
   - Contains: `DistL1`, `DistL2`, `DistCosine`, `DistDot` with all implementations and macros
   - Commits: `a9c95d8`

4. **`src/dist/probability.rs`** ✅
   - Status: Complete and tested
   - Contains: `DistHellinger`, `DistJeffreys`, `DistJensenShannon` with all implementations and macros
   - Commits: `080dbb9`

5. **`src/dist/set.rs`** ✅
   - Status: Complete and tested
   - Contains: `DistHamming`, `DistJaccard` with all implementations and macros
   - Commits: `86b99ab`

6. **`src/dist/string.rs`** ✅
   - Status: Complete and tested
   - Contains: `DistLevenshtein` implementation
   - Commits: `b0ec1fd`

7. **`src/dist/custom.rs`** ✅
   - Status: Complete and tested
   - Contains: `NoDist`, `DistFn`, `DistPtr`, `DistCFFI`, `DistCFnPtr` type alias
   - Commits: `d532425`

### 🔄 In Progress
- None currently

### ⏳ Remaining Modules
8. `src/dist/unifrac.rs` - All UniFrac implementations (largest module)
9. Final cleanup - Remove old `distances.rs` file and verify everything works

## Proposed Module Structure

### 1. `src/dist/traits.rs` (~50 lines) ✅ COMPLETE
**Purpose**: Core trait definition and common types

**Contents**:
- `Distance<T>` trait definition
- Documentation for the trait

**Dependencies**: None (foundation)

**Status**: ✅ Extracted and tested. Note: Renamed from `trait.rs` to `traits.rs` because `trait` is a Rust keyword.

---

### 2. `src/dist/basic.rs` (~400 lines) ✅ COMPLETE
**Purpose**: Basic vector distance metrics (L1, L2, Cosine, Dot)

**Contents**:
- `DistL1` struct and implementations
- `DistL2` struct and implementations  
- `DistCosine` struct and implementations
- `DistDot` struct and implementations
- Macros: `implementL1Distance!`, `implementL2Distance!`, `implementCosDistance!`, `implementDotDistance!`
- Helper functions: `scalar_l2_f32()`, `scalar_dot_f32()`

**Dependencies**:
- `traits.rs` (for `Distance` trait)
- `distsimd.rs` (for SIMD implementations)
- `disteez.rs` (for SIMD implementations)

**Status**: ✅ Extracted and tested. All basic distance implementations moved successfully.

---

### 3. `src/dist/probability.rs` (~200 lines) ✅ COMPLETE
**Purpose**: Probability distribution distances

**Contents**:
- `DistHellinger` struct and implementations
- `DistJeffreys` struct and implementations
- `DistJensenShannon` struct and implementations
- Macros: `implementHellingerDistance!`, `implementJeffreysDistance!`, `implementDistJensenShannon!`
- Constant: `M_MIN` for Jeffreys distance

**Dependencies**:
- `traits.rs` (for `Distance` trait)
- `disteez.rs` (for SIMD implementations)

**Status**: ✅ Extracted and tested. All probability distance implementations moved successfully.

---

### 4. `src/dist/set.rs` (~200 lines) ✅ COMPLETE
**Purpose**: Set-based distance metrics

**Contents**:
- `DistHamming` struct and implementations (for multiple types: u8, u16, u32, u64, i16, i32, f32, f64)
- `DistJaccard` struct and implementations (for u8, u16, u32)
- Macros: `implementHammingDistance!`, `implementJaccardDistance!`

**Dependencies**:
- `traits.rs` (for `Distance` trait)
- `distsimd.rs` (for SIMD implementations)
- `disteez.rs` (for SIMD implementations)

**Status**: ✅ Extracted and tested. All set distance implementations moved successfully.

---

### 5. `src/dist/string.rs` (~60 lines) ✅ COMPLETE
**Purpose**: String distance metrics

**Contents**:
- `DistLevenshtein` struct and implementation (for u16)

**Dependencies**:
- `traits.rs` (for `Distance` trait)

**Status**: ✅ Extracted and tested. Levenshtein distance implementation moved successfully.

---

### 6. `src/dist/unifrac.rs` (~1,200 lines)
**Purpose**: UniFrac phylogenetic distance implementations

**Contents**:
- `DistUniFrac` struct and implementation
- `NewDistUniFrac` struct and implementation
- `DistUniFracCFFI` struct and implementation (if not commented out)
- `DistUniFrac_C` struct and related FFI types
- `OpaqueBPTree` struct
- `ComputeStatus` enum
- Helper functions:
  - `build_tint_lint()`
  - `build_leaf_map()`
  - `extract_leaf_names_from_newick_string()`
  - `unifrac_succparen_normalized()`
  - `unifrac_pair()`
  - `collect_children()`
  - `make_presence_mask_f32_scalar()`
  - `make_presence_mask_f32()`
  - `or_masks()`
  - `extract_set_bits()`
  - `compute_unifrac_for_pair_unweighted_bitwise()`
  - `compute_unifrac_for_pair_weighted_bitwise()`
- FFI functions for C interop

**Dependencies**:
- `trait.rs`
- External crates: `phylotree`, `bitvec`, `succparen`, `newick`, `anyhow`

**Line ranges**: ~700-1864

**Note**: This is the largest module and could potentially be split further into:
- `unifrac/mod.rs` - Main exports
- `unifrac/original.rs` - `DistUniFrac` implementation
- `unifrac/new.rs` - `NewDistUniFrac` implementation  
- `unifrac/ffi.rs` - C FFI implementations
- `unifrac/utils.rs` - Helper functions

---

### 7. `src/dist/custom.rs` (~150 lines) ✅ COMPLETE
**Purpose**: Custom distance wrappers and adapters

**Contents**:
- `NoDist` struct and implementation
- `DistFn<T>` struct and implementation (closure-based)
- `DistPtr<T, F>` struct and implementation (function pointer)
- `DistCFFI<T>` struct and implementation (C function pointer)
- `DistCFnPtr<T>` type alias

**Dependencies**:
- `traits.rs` (for `Distance` trait)
- `num_traits::Float` (for `DistPtr`)
- `std::os::raw::c_ulonglong` (for C FFI)

**Status**: ✅ Extracted and tested. All custom distance implementations moved successfully.

---

### 8. `src/dist/utils.rs` (~50 lines) ✅ COMPLETE
**Purpose**: Utility functions used across distance implementations

**Contents**:
- `l2_normalize()` function
- Other shared utility functions (if any)

**Dependencies**: None (or minimal)

**Status**: ✅ Extracted and tested. `l2_normalize()` function moved successfully.

---

### 9. `src/dist/macros.rs` (optional, ~200 lines)
**Purpose**: Shared macros for repetitive implementations

**Alternative**: Keep macros in their respective modules (recommended)

**If created, contents**:
- All `implement*Distance!` macros
- Shared macro utilities

---

## Updated `src/dist/mod.rs` Structure

**Current State** (as of latest commit):
```rust
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
```

**Target Structure** (when complete):
```rust
//! Distance implementations module

// Core trait
pub mod traits;
pub use traits::Distance;

// Utilities
pub mod utils;

// Distance implementations
pub mod basic;
pub mod probability;
pub mod set;
pub mod string;
pub mod unifrac;
pub mod custom;

// SIMD implementations (existing)
pub(crate) mod distsimd;
pub(crate) mod disteez;

// Re-export all distance types
pub use basic::*;
pub use probability::*;
pub use set::*;
pub use string::*;
pub use unifrac::*;
pub use custom::*;
pub use utils::*;
```

---

## Migration Strategy

### ✅ Phase 1: Extract Core Components - COMPLETE
1. ✅ Create `traits.rs` with `Distance` trait (renamed from `trait.rs`)
2. ✅ Create `utils.rs` with utility functions
3. ✅ Update `mod.rs` to include new modules
4. ✅ Update `prelude.rs` to use re-exported trait

### ✅ Phase 2: Extract Basic Distances - COMPLETE
1. ✅ Create `basic.rs` with L1, L2, Cosine, Dot
2. ✅ Move macros and implementations
3. ✅ Update imports in `distances.rs`
4. ✅ All tests pass

### ✅ Phase 3: Extract Specialized Distances - COMPLETE
1. ✅ Create `probability.rs` (Hellinger, Jeffreys, Jensen-Shannon)
2. ✅ Create `set.rs` (Hamming, Jaccard)
3. ✅ Create `string.rs` (Levenshtein)
4. ✅ Update imports and remove duplicate code
5. ✅ All tests pass

### ✅ Phase 4: Extract Complex Modules - MOSTLY COMPLETE
1. ✅ Create `custom.rs` (NoDist, DistFn, DistPtr, DistCFFI)
2. ⏳ Create `unifrac.rs` (largest, most complex) - **NEXT**
3. ⏳ Move all remaining implementations

### ⏳ Phase 5: Cleanup - PENDING
1. ⏳ Remove old `distances.rs` file
2. ⏳ Update all internal imports
3. ⏳ Verify tests still pass
4. ⏳ Update documentation
5. ⏳ Split tests into respective modules (optional)

---

## File Size Estimates

| Module | Estimated Lines | Complexity | Status |
|--------|----------------|------------|--------|
| `traits.rs` | ~50 | Low | ✅ Complete |
| `basic.rs` | ~400 | Medium | ✅ Complete |
| `probability.rs` | ~200 | Medium | ✅ Complete |
| `set.rs` | ~200 | Medium | ✅ Complete |
| `string.rs` | ~60 | Low | ✅ Complete |
| `unifrac.rs` | ~1,200 | High | ⏳ Pending |
| `custom.rs` | ~150 | Low | ✅ Complete |
| `utils.rs` | ~50 | Low | ✅ Complete |
| **Total** | **~2,310** | | **7/8 Complete** |

*Note: Some reduction expected due to shared imports and better organization*

**Actual Sizes** (as extracted):
- `traits.rs`: ~35 lines
- `utils.rs`: ~30 lines  
- `basic.rs`: ~350 lines
- `probability.rs`: ~167 lines
- `set.rs`: ~163 lines
- `string.rs`: ~58 lines
- `custom.rs`: ~114 lines

---

## Benefits

1. **Maintainability**: Each module has a clear, focused purpose
2. **Discoverability**: Easier to find specific distance implementations
3. **Compilation**: Faster incremental compilation (only changed modules rebuild)
4. **Testing**: Can test modules independently
5. **Documentation**: Better organization for documentation generation
6. **Collaboration**: Multiple developers can work on different modules simultaneously

---

## Test Organization

**Current**: All tests are in a single `#[cfg(test)]` module at the end of `distances.rs` (~2,025 lines)

**Options**:
1. **Keep tests in each module** (recommended)
   - Each module has its own `#[cfg(test)]` section
   - Tests live alongside the code they test
   - Easier to maintain and understand

2. **Create separate test modules**
   - `src/dist/tests/` directory
   - `basic_tests.rs`, `probability_tests.rs`, etc.
   - Keeps implementation files cleaner

3. **Integration tests**
   - Move to `tests/` directory at crate root
   - Test the public API only

**Recommendation**: Option 1 - Keep tests in their respective modules using `#[cfg(test)]` blocks. This follows Rust conventions and keeps related code together.

---

## Considerations

1. **Circular Dependencies**: ✅ No issues encountered. Proper module organization maintained
2. **Public API**: ✅ Backward compatibility maintained - all types accessible via `dist::*` through re-exports
3. **Tests**: ✅ Tests still in `distances.rs` but working correctly. Can move to respective modules later (optional)
4. **SIMD Dependencies**: ✅ Feature gates working correctly - all modules successfully use conditional SIMD compilation
5. **UniFrac Complexity**: ⏳ Next module to extract. Consider further splitting if it remains too large (~1,200 lines)
6. **Test Migration**: ⏳ Tests currently work from `distances.rs` with imports. Can migrate later for better organization
7. **Rust Keywords**: ✅ Note that `trait` is a Rust keyword, so module was named `traits.rs` instead
8. **Import Organization**: ✅ Using `use super::module::*` pattern for internal imports works well
9. **Duplicate Code Removal**: ✅ Successfully removed duplicate implementations from `distances.rs` after extraction
10. **Test Imports**: ✅ Added necessary imports in test module (e.g., `use super::super::set::{DistHamming, DistJaccard}`)

---

## Alternative: Further Split UniFrac Module

If `unifrac.rs` is still too large (~1,200 lines), consider:

```
src/dist/unifrac/
  mod.rs          # Re-exports and module declarations
  original.rs     # DistUniFrac (~250 lines)
  new.rs          # NewDistUniFrac (~600 lines)
  ffi.rs          # DistUniFracCFFI and C interop (~200 lines)
  utils.rs        # Helper functions (~150 lines)
```

This would make the UniFrac implementations more manageable while keeping related code together.
