# Utilities

This directory contains utility scripts and examples for the anndists crate.

## Scripts

- **check_data.rs** - Quick utility to verify data files are readable
- **validate_data.rs** - Validates mouse gut microbiome data files
- **compare_output.rs** - Example showing different ways to use the Distance trait
- **distance_trait_example.rs** - Example demonstrating Distance trait usage

## Usage

These scripts are not configured as Cargo binaries. To run them:

```bash
# From the project root
rustc utilities/check_data.rs --edition 2021 --extern anndists=target/debug/libanndists.rlib && ./check_data

# Or compile and run manually
cd utilities
rustc check_data.rs --edition 2021 -L ../target/debug/deps && ./check_data
```

Note: These scripts reference data files using relative paths (`../data/`), so they should be run from the project root or the utilities directory depending on the path structure.
