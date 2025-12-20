// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Estimation functions for the Powerlaw distribution.

/// Calculates the Maximum Likelihood Estimate (MLE) for the alpha parameter of a power-law distribution.
///
/// This estimates the power-law exponent `alpha` (where f(x) ~ x^-alpha).
///
/// # Parameters
/// - `data`: A slice of `f64` values representing the data points. It's assumed that all `x` in `data` are `>= x_min`.
/// - `x_min`: The minimum value of the distribution.
///
/// # Returns
/// The estimated alpha parameter as an `f64`.
/// Note: This value is equivalent to `1 + alpha_pareto`, where `alpha_pareto` is the shape parameter
/// of the standard Pareto Type I distribution.
pub fn alpha_hat(data: &[f64], x_min: f64) -> f64 {
    let n: usize = data.len();

    let logs = data.iter().map(|x: &f64| (x / x_min).ln());
    let sum_of_logs: f64 = logs.sum();

    1. + n as f64 / sum_of_logs
}
