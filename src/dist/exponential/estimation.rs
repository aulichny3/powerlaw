// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Estimation functions for the Exponential distribution.

/// Calculates the Maximum Likelihood Estimate (MLE) for the lambda parameter of an exponential distribution.
///
/// # Parameters
/// - `data`: A slice of `f64` values representing the data points. It's assumed that all `x` in `data` are `>= x_min`.
/// - `x_min`: The minimum value of the distribution.
///
/// # Returns
/// The estimated lambda parameter as an `f64`.
pub fn lambda_hat(data: &[f64], x_min: f64) -> f64 {
    // This is a placeholder. The user should implement the actual MLE calculation.
    // For a shifted exponential distribution, the MLE for lambda is:
    // n / sum(x_i - x_min)
    // where n is the number of data points.
    let srt: Vec<_> = data.into_iter().filter(|&x| x >= &x_min).collect();
    let n = srt.len() as f64;

    n / srt.iter().map(|&x| x - x_min).sum::<f64>()
}
