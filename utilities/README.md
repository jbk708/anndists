# Utilities

This directory contains utility scripts and examples for the anndists crate.

## Scripts

- **check_data.rs** - Quick utility to verify data files are readable
- **validate_data.rs** - Validates mouse gut microbiome data files
- **compare_output.rs** - Example showing different ways to use the Distance trait
- **distance_trait_example.rs** - Example demonstrating Distance trait usage

## Usage

These scripts are not configured as Cargo binaries. They reference data files using relative paths (`../data/`), so they should be run from the project root directory.

To compile and run them manually:

```bash
# From the project root
cd utilities
rustc check_data.rs --edition 2021 -L ../target/debug/deps --extern anndists=../target/debug/libanndists.rlib && ./check_data
```

Or, if you want to run from the project root:

```bash
# From the project root
rustc utilities/check_data.rs --edition 2021 -L target/debug/deps --extern anndists=target/debug/libanndists.rlib -o utilities/check_data && utilities/check_data
```

**Note**: These are utility scripts for development/debugging purposes. For production use, prefer the examples in the `examples/` directory which are properly configured as Cargo examples.
