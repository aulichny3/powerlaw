// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! powerlaw is a library with a cli developed to assist in parameter estimation and hypothesis testing of Power Law distributed data.
//! # Project Goals
//! The goal of powerlaw is to offer a highly performant library to aid in the study of Power Law distributed data, a subclass of probability distributions
//! characterized by their heavy tails. Such distributions are of interest in numerous fields of study within natural science, social science, and formal science.
//! # Installation
//! ```bash
//! cargo install powerlaw
//! ```
//! # Requirements
//! One column of csv data if using from the command line (the file extension does not matter). Otherwise one ```Vec<f64>``` of the data if using the library functions. Other file formats/data types are not yet supported.
//! # Usage

//!
//! The `powerlaw` CLI tool provides two main subcommands: `fit` and `test`.
//!
//! ### `fit` subcommand
//!
//! Use the `fit` subcommand to perform the initial analysis, finding the maximum likelihood estimates
//! for the `x_min` and `alpha` parameters of a power-law distribution. This command does not perform
//! the computationally intensive hypothesis test.
//!
//! ```bash
//! # Example: Fit a Power Law to data in 'Data/reference_data/blackouts.txt '
//! cargo run -- fit Data/reference_data/blackouts.txt
//!```
//!
//! ### `test` subcommand
//!
//! Use the `test` subcommand to perform the full analysis, including the hypothesis test to determine
//! if the data is plausibly drawn from a power-law distribution. This command requires a `--precision`
//! argument to specify the desired accuracy for the p-value calculation.
//!
//! ```bash
//! # Example: Test data in 'Data/reference_data/blackouts.txt ' with a precision of 0.01
//! cargo run -- test Data/reference_data/blackouts.txt  --precision 0.01
//!
//! # You can also use the short flag for precision:
//! cargo run -- test Data/reference_data/blackouts.txt  -p 0.01
//!```
//!
//! ### Getting Help
//!
//! `clap` automatically generates comprehensive help messages. You can view them by running:
//!
//! ```bash
//! # General help for the powerlaw CLI
//! cargo run -- --help
//!
//! # Help for a specific subcommand, e.g., 'test'
//! cargo run -- test --help
//! ```
//!
//! where "0.1" in this example is the decimal precision desired during hypothesis testing see [util::sim::calculate_sim_params] for more details.
//! # Methodology
//! The vast majority of this package is based on the procedure outlined in ‘Power-Law Distributions in Empirical Data’ by Clauset et al.
//! # Limitations
//! 1. Only the continuous case of the Pareto Type I Distribution is considered for parameter estimation, goodness of fit, and hypothesis testing at this time. This may or may not change with future updates. The example data in the documentation is discrete, thus the results are only an approximation.
//! 2. Domain knowledge of the data generating process is critical given the methodology used by this package is based on that proposed by the referenced material.
//! Specifically the 1 sample Kolmogorov-Smirnov test is used for goodness of fit testing which assumes i.i.d data. Many natural processes data are serially correlated, thus KS testing is not appropriate, see references section below.
//! 3. This is highly alpha code; backwards compatibility is not guaranteed and should not be expected.
//! 4. Many more known and unknown.
//! # References
//! Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-Law Distributions in Empirical Data. SIAM Review, 51(4), 661–703. [doi:10.1137/070710111](https://doi.org/10.1137/070710111)
//!
//! Jeff Alstott, Ed Bullmore, Dietmar Plenz. (2014). powerlaw: a Python package for analysis of heavy-tailed distributions. PLoS ONE 9(1): e85777 [doi:10.1371/journal.pone.0085777](http://dx.doi.org/10.1371/journal.pone.0085777)
//!
//! Zeimbekakis, A., Schifano, E. D., & Yan, J. (2024). On Misuses of the Kolmogorov-Smirnov Test for One-Sample Goodness-of-Fit. The American Statistician, 78(4), 481–487. [10.1080/00031305.2024.2356095](https://doi.org/10.1080/00031305.2024.2356095)

pub mod dist;
pub mod stats;
pub mod util;

// Re-export the trait
pub use dist::Distribution;
