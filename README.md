# Powerlaw

[![Crates.io](https://img.shields.io/crates/v/powerlaw.svg)](https://crates.io/crates/powerlaw)
[![Docs.rs](https://docs.rs/powerlaw/badge.svg)](https://docs.rs/powerlaw)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](./LICENSE-MIT)

A Rust library and command-line tool for analyzing power-law distributions in empirical data.

## Overview

`powerlaw` is a high-performance Rust library developed to assist in parameter estimation and hypothesis testing of power-law distributed data. Such distributions are of interest in numerous fields of study, from natural to social sciences.

The methodology is heavily based on the techniques and statistical framework described in the paper ['Power-Law Distributions in Empirical Data'](https://doi.org/10.1137/070710111) by Aaron Clauset, Cosma Rohilla Shalizi, and M. E. J. Newman.

## Features

-   **Parameter Estimation**: Estimates the parameters (`x_min`, `alpha`) of a power-law distribution from data.
-   **Goodness-of-Fit**: Uses the Kolmogorov-Smirnov (KS) statistic to find the best-fitting parameters.
-   **Hypothesis Testing**: Performs a hypothesis test to determine if the power-law model is a plausible fit for the data.
-   **High Performance**: Computationally intensive tasks are parallelized using Rayon for significant speedups.
-   **Dual Use**: Can be used as a simple command-line tool or as a library in other Rust projects.

## Requirements
Rust 2021 or greater.

## Installation

You can install the CLI tool directly from the Git repository:

```bash
cargo install --git https://github.com/aulichny3/powerlaw.git
```

Or from [crates.io](https://crates.io):

```bash
cargo install powerlaw
```
![installation demo gif](.github/install.gif)

## CLI Usage

The `powerlaw` CLI provides two main subcommands: `fit` and `test`.

### `fit` subcommand

Use `fit` to perform the initial analysis, finding the maximum likelihood estimates for the `x_min` and `alpha` parameters. This command does not perform the computationally intensive hypothesis test.

**Command:**

```bash
powerlaw fit <FILEPATH>
```
![Fit demo gif](.github/fit.gif)
**Example:**

```
$ powerlaw fit Data/reference_data/blackouts.txt

Data: ./Data/reference_data/blackouts.txt
n: 211
-- Pareto Type I parameters -- 
alpha:          1.2726372198302858 
x_min:          230000.0 
KS stat:        0.06067379629443781 
tail length:    59

-- Generic Power-Law [Cx^(-alpha)] parameters -- 
alpha:          2.272637219830286 
x_min:          230000.0 
KS stat:        0.06067379629443781 
tail length:    59
```

### `test` subcommand

Use `test` to perform the full analysis, including the hypothesis test to determine if the data is plausibly drawn from a power-law distribution. This command requires a `--precision` argument for the p-value calculation.
**Caution: This function can be very slow depending on the data**.


**Command:**

```bash
powerlaw test <FILEPATH> --precision <VALUE>
```
![Hypothesis test demo gif](.github/test.gif)
**Example:**

```
$ powerlaw test Data/reference_data/blackouts.txt --precision 0.01

Data: ./Data/reference_data/blackouts.txt
Precision: 0.01
n: 211
-- Pareto Type I parameters -- 
alpha:          1.2726372198302858 
x_min:          230000.0 
KS stat:        0.06067379629443781 
tail length:    59

-- Generic Power-Law [Cx^(-alpha)] parameters -- 
alpha:          2.272637219830286 
x_min:          230000.0 
KS stat:        0.06067379629443781 
tail length:    59

-- Parameter Uncertainty --
x_min std:      93081.20130389203 
alpha std:      0.26478692708043355

-- Hypothesis Testing --
Generating M = 2500 simulated datasets of length n = 211 with tail size 59 and probability of the tail P(tail|data) = 0.2796208530805687
Qty of simulations with KS statistic > empirical data = 1963
p-value: 0.7852
Fail to reject the null H0: Power-Law distribution is a plausible fit to the data.
```

### Getting Help
```bash
# General help
powerlaw --help

# Help for a specific subcommand
powerlaw test --help
```

## Library Usage

You can also use `powerlaw` as a library in your own Rust projects.

**1. Add to `Cargo.toml`:**

```toml
[dependencies]
powerlaw = "0.0.16" # Or the version you need
```

**2. Example:**

```rust
use powerlaw::{dist, util};

fn main() {
    // 1. Read your data into a Vec<f64>
    let mut data = util::read_csv("path/to/your/data.csv").unwrap();

    // 2. Find the MLE alphas for all potential x_mins
    let alphas = dist::pareto::find_alphas_fast(&mut data);

    // 3. Find the best fit (x_min, alpha) pair based on the KS statistic
    let best_fit = dist::pareto::gof(&data, &alphas.0, &alphas.1);

    println!("Best fit found: x_min = {}, alpha = {}", best_fit.x_min, best_fit.alpha);

    // 4. Optionally, run the hypothesis test
    let precision = 0.01;
    let h_0 = dist::pareto::hypothesis_test(
        data,
        precision,
        best_fit.alpha,
        best_fit.x_min,
        best_fit.D,
    );

    println!("Hypothesis test p-value: {}", h_0.p);
    if h_0.p > 0.1 {
        println!("The power-law model is a plausible fit.");
    } else {
        println!("The power-law model is not a plausible fit.");
    }
}
```

## Building from Source

```bash
# Clone the repository
git clone https://github.com/aulichny3/powerlaw
cd powerlaw

# Build in release mode
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## Limitations
1. Only the continuous case of the Pareto Type I Distribution is considered for parameter estimation, goodness of fit, and hypothesis testing at this time. This may or may not change with future updates. The example data in the documentation is discrete, thus the results are only an approximation.
2. Domain knowledge of the data generating process is critical given the methodology used by this package is based on that proposed by the referenced material.
Specifically the 1 sample Kolmogorov-Smirnov test is used for goodness of fit testing which assumes i.i.d data. Many natural processes data are serially correlated, thus KS testing is not appropriate, see references section below.
3. This is highly alpha code; backwards compatibility is not guaranteed and should not be expected.
4. Many more known and unknown.

## License

This project is licensed under either of

-   Apache License, Version 2.0, ([LICENSE-APACHE](./LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
-   MIT license ([LICENSE-MIT](./LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
