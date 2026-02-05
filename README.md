# Powerlaw

[![Crates.io](https://img.shields.io/crates/v/powerlaw.svg)](https://crates.io/crates/powerlaw)
[![Docs.rs](https://docs.rs/powerlaw/badge.svg)](https://docs.rs/powerlaw)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](./LICENSE-MIT)

A Rust library and command-line tool for analyzing power-law distributions in empirical data.

## Overview

`powerlaw` is the high-performance Rust library backend of the Python package [powerlawrs](https://github.com/aulichny3/powerlawrs) developed to assist in parameter estimation and hypothesis testing of power-law distributed data. Such distributions are of interest in numerous fields of study, from natural to social sciences.

The methodology is heavily based on the techniques and statistical framework described in the paper ['Power-Law Distributions in Empirical Data'](https://doi.org/10.1137/070710111) by Aaron Clauset, Cosma Rohilla Shalizi, and M. E. J. Newman.

## Features

-   **Parameter Estimation**: Estimates the parameters (`x_min`, `alpha`) of a power-law distribution from data.
-   **Goodness-of-Fit**: Uses the Kolmogorov-Smirnov (KS) statistic to find the best-fitting parameters.
-   **Hypothesis Testing**: Performs a hypothesis test to determine if the power-law model is a plausible fit for the data.
-   **Vuongs Closeness Test**: Model selection for non-nested models.
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
![installation demo gif](https://raw.githubusercontent.com/aulichny3/powerlaw/main/.github/install.gif)

## CLI Usage

The `powerlaw` CLI provides two main subcommands: `fit` and `test`.

### `fit` subcommand

Use `fit` to perform the initial analysis, finding the maximum likelihood estimates for the `x_min` and `alpha` parameters. This command does not perform the computationally intensive hypothesis test.

**Command:**

```bash
powerlaw fit <FILEPATH>
```
![Fit demo gif](https://raw.githubusercontent.com/aulichny3/powerlaw/main/.github/fit.gif)
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
![Hypothesis test demo gif](https://raw.githubusercontent.com/aulichny3/powerlaw/main/.github/test.gif)
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

-- Vuongs Closeness Test --
Z score: 1.4208153737166338
p-value: 0.15537054041363918
No significant difference between Pareto and Exponential models.
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
powerlaw = "0.0.24" # Or the latest version
```

**2. Example: Fitting and Comparing Distributions**

The new API makes it easy to find the best-fit `x_min` for a power-law distribution and then compare it against other distributions that are fit to the same tail portion of the data.

```rust
use powerlaw::{
    dist::{self, exponential::Exponential, pareto::Pareto}, 
    util, 
    FittedResults,
};

fn main() {
    // 1. Read your data into a Vec<f64>
    let mut data = util::read_csv("path/to/your/data.csv").unwrap();

    // 2. Find the best-fit Pareto Type I parameters to determine the tail of the distribution
    let (x_mins, alphas) = dist::pareto::find_alphas_fast(&mut data);
    let pareto_fit = dist::pareto::gof(&data, &x_mins, &alphas);
    println!(
        "Pareto Type I best fit found: x_min = {}, alpha = {}",
        pareto_fit.x_min, pareto_fit.alpha
    );

    // 3. Create a manager to hold the results of fitting other distributions for comparison later.
    let mut results = FittedResults::new();

    // 4. Create fully-formed distribution objects from the initial pareto_fit
    //    and add them to the results manager.

    // Create a Pareto distribution object from the pareto_fit struct for access to pdf(), cdf() etc  
    let pareto_dist = Pareto::from(pareto_fit.clone());
    results.add(pareto_dist);

    // Create an Shifted Exponential distribution object using the x_min from the pareto_fit
    let exp_dist = Exponential::from_fitment(&data, &pareto_fit);
    results.add(exp_dist);

    // (In the future, you could add more distributions here, e.g., Student T etc.)

    // 5. Get a summary of all fitted distributions
    println!("\n--- Fitted Distributions Summary ---");
    let summary = results.summary();
    for (name, params) in summary {
        println!("Distribution: {}", name);
        for (param_name, param_value) in params {
            println!("  - {}: {}", param_name, param_value);
        }
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
1. Only the continuous case of the Pareto Type I Distribution is considered for parameter estimation, goodness of fit, and hypothesis testing conditional on the sample distribution. 
2. Parameter estimation of other distributions (shifted exponential, lognormal, etc.) is conditional on the tail found during the Pareto Type I model selection step. 
3. The example data in the documentation is discrete, thus the results are only an approximation.
4. Domain knowledge of the data generating process is critical given the methodology used by this package is based on that proposed by the referenced material.
specifically the 1 sample Kolmogorov-Smirnov test is used for goodness of fit testing which assumes i.i.d data. Many natural processes data are serially correlated, thus KS testing is not appropriate, see references section below.
5. This is highly alpha code; backwards compatibility is not guaranteed and should not be expected. 

## License

This project is licensed under either of

-   Apache License, Version 2.0, ([LICENSE-APACHE](./LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
-   MIT license ([LICENSE-MIT](./LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## References
Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-Law Distributions in Empirical Data. SIAM Review, 51(4), 661–703. [doi:10.1137/070710111](https://doi.org/10.1137/070710111)

Jeff Alstott, Ed Bullmore, Dietmar Plenz. (2014). powerlaw: a Python package for analysis of heavy-tailed distributions. PLoS ONE 9(1): e85777 [doi:10.1371/journal.pone.0085777](http://dx.doi.org/10.1371/journal.pone.0085777)

Zeimbekakis, A., Schifano, E. D., & Yan, J. (2024). On Misuses of the Kolmogorov-Smirnov Test for One-Sample Goodness-of-Fit. The American Statistician, 78(4), 481–487. [10.1080/00031305.2024.2356095](https://doi.org/10.1080/00031305.2024.2356095)