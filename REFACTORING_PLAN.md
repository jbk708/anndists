# Refactoring Plan: Breaking up `distances.rs`

## Current State
- **File**: `src/dist/distances.rs`
- **Size**: ~3,935 lines total
  - Implementation code: ~1,910 lines
  - Test code: ~2,025 lines (lines 1911-3935)
- **Structure**: Single monolithic file containing all distance implementations and tests

## Proposed Module Structure

### 1. `src/dist/trait.rs` (~50 lines)
**Purpose**: Core trait definition and common types

**Contents**:
- `Distance<T>` trait definition
- `DistKind` enum (if still needed)
- Common imports and type aliases
- Documentation for the trait

**Dependencies**: None (foundation)

---

### 2. `src/dist/basic.rs` (~400 lines)
**Purpose**: Basic vector distance metrics (L1, L2, Cosine, Dot)

**Contents**:
- `DistL1` struct and implementations
- `DistL2` struct and implementations  
- `DistCosine` struct and implementations
- `DistDot` struct and implementations
- Macros: `implementL1Distance!`, `implementL2Distance!`, `implementCosDistance!`, `implementDotDistance!`
- Helper functions: `scalar_l2_f32()`, `scalar_dot_f32()`

**Dependencies**:
- `trait.rs` (for `Distance` trait)
- `distsimd.rs` (for SIMD implementations)
- `disteez.rs` (for SIMD implementations)

**Line ranges**: ~99-339

---

### 3. `src/dist/probability.rs` (~200 lines)
**Purpose**: Probability distribution distances

**Contents**:
- `DistHellinger` struct and implementations
- `DistJeffreys` struct and implementations
- `DistJensenShannon` struct and implementations
- Macros: `implementHellingerDistance!`, `implementJeffreysDistance!`, `implementDistJensenShannon!`

**Dependencies**:
- `trait.rs`
- `disteez.rs` (for SIMD implementations)

**Line ranges**: ~339-492

---

### 4. `src/dist/set.rs` (~200 lines)
**Purpose**: Set-based distance metrics

**Contents**:
- `DistHamming` struct and implementations
- `DistJaccard` struct and implementations
- Macros: `implementHammingDistance!`, `implementJaccardDistance!`

**Dependencies**:
- `trait.rs`
- `distsimd.rs` (for SIMD implementations)
- `disteez.rs` (for SIMD implementations)

**Line ranges**: ~492-647

---

### 5. `src/dist/string.rs` (~60 lines)
**Purpose**: String distance metrics

**Contents**:
- `DistLevenshtein` struct and implementation

**Dependencies**:
- `trait.rs`

**Line ranges**: ~647-700

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

### 7. `src/dist/custom.rs` (~150 lines)
**Purpose**: Custom distance wrappers and adapters

**Contents**:
- `NoDist` struct and implementation
- `DistFn<T>` struct and implementation (closure-based)
- `DistPtr<T, F>` struct and implementation (function pointer)
- `DistCFFI<T>` struct and implementation (C function pointer)
- `DistCFnPtr<T>` type alias

**Dependencies**:
- `trait.rs`
- `num_traits::Float` (for `DistPtr`)

**Line ranges**: ~86-98, ~1620-1909

---

### 8. `src/dist/utils.rs` (~50 lines)
**Purpose**: Utility functions used across distance implementations

**Contents**:
- `l2_normalize()` function
- Other shared utility functions

**Dependencies**: None (or minimal)

**Line ranges**: ~330

---

### 9. `src/dist/macros.rs` (optional, ~200 lines)
**Purpose**: Shared macros for repetitive implementations

**Alternative**: Keep macros in their respective modules (recommended)

**If created, contents**:
- All `implement*Distance!` macros
- Shared macro utilities

---

## Updated `src/dist/mod.rs` Structure

```rust
//! Distance implementations module

// Core trait
pub mod trait;
pub use trait::Distance;

// Distance implementations
pub mod basic;
pub mod probability;
pub mod set;
pub mod string;
pub mod unifrac;
pub mod custom;

// Utilities
pub mod utils;

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

### Phase 1: Extract Core Components
1. Create `trait.rs` with `Distance` trait
2. Create `utils.rs` with utility functions
3. Update `mod.rs` to include new modules

### Phase 2: Extract Basic Distances
1. Create `basic.rs` with L1, L2, Cosine, Dot
2. Move macros and implementations
3. Update imports in `distances.rs` temporarily

### Phase 3: Extract Specialized Distances
1. Create `probability.rs`
2. Create `set.rs`
3. Create `string.rs`
4. Update imports

### Phase 4: Extract Complex Modules
1. Create `unifrac.rs` (largest, most complex)
2. Create `custom.rs`
3. Move all remaining implementations

### Phase 5: Cleanup
1. Remove old `distances.rs` file
2. Update all internal imports
3. Verify tests still pass
4. Update documentation

---

## File Size Estimates

| Module | Estimated Lines | Complexity |
|--------|----------------|------------|
| `trait.rs` | ~50 | Low |
| `basic.rs` | ~400 | Medium |
| `probability.rs` | ~200 | Medium |
| `set.rs` | ~200 | Medium |
| `string.rs` | ~60 | Low |
| `unifrac.rs` | ~1,200 | High |
| `custom.rs` | ~150 | Low |
| `utils.rs` | ~50 | Low |
| **Total** | **~2,310** | |

*Note: Some reduction expected due to shared imports and better organization*

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

1. **Circular Dependencies**: Ensure proper module organization to avoid cycles
2. **Public API**: Maintain backward compatibility - all types should still be accessible via `dist::*`
3. **Tests**: Move tests to respective modules (recommended) or keep in integration tests
4. **SIMD Dependencies**: Some modules depend on `distsimd.rs` and `disteez.rs` - ensure feature gates work correctly
5. **UniFrac Complexity**: Consider further splitting `unifrac.rs` if it remains too large
6. **Test Migration**: Tests will need to be split and moved alongside their implementations

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
