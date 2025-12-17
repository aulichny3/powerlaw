// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Correctness and validation test suite for the powerlaw crate.
//!
//! This module validates the statistical algorithms against known "Gold Standard" results
//! derived from statistical literature or trusted software packages (e.g., R, scipy).
//! It focuses on ensuring the accuracy of:
//! 1. Empirical CDF calculations.
//! 2. Parameter Estimation (MLE) for various distributions (especially truncated ones).
//! 3. Kolmogorov-Smirnov (KS) statistic calculations.
//! 4. Log-likelihood calculations.

use powerlaw::{
    dist::{self, Distribution},
    stats,
};

#[test]
fn test_ecdf_calculation() {
    // Action Item 1 from Phase 0: Validate ECDF Calculation.
    // Standard ECDF definition: F_n(x) = (number of elements <= x) / n
    // However, for KS distance relative to a fitted distribution on x >= x_min,
    // care must be taken with the definition used by Clauset et al.

    // TODO: Implement specific test cases comparing our ECDF logic against manual calculations.
}

#[test]
fn test_pareto_mle_correctness() {
    // Validation of Power-Law alpha Estimation (MLE)
    // Reference: Clauset et al. (2009)

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x_min = 1.0;

    // Manual calculation or known result for this simple dataset
    // alpha = 1 + n / sum(ln(x/x_min))
    // sum_ln = ln(1)+ln(2)+...+ln(10) = ln(3628800) approx 15.1044
    // n = 10
    // alpha = 1 + 10 / 15.1044 = 1 + 0.66205 = 1.66205

    let alpha = dist::pareto::estimation::type1_alpha_hat(&data, x_min, false);

    assert!(
        (alpha - 0.66205).abs() < 0.0001,
        "Pareto MLE alpha estimation failed validation. Expected ~0.66205, got {}",
        alpha
    );
}

#[test]
fn test_exponential_mle_correctness() {
    // Validation of Exponential MLE on truncated data.
    // MLE for shifted exponential: lambda = n / sum(x_i - x_min)

    let data = vec![2.0, 2.5, 3.0, 3.5, 4.0];
    let x_min = 2.0;
    // x - x_min: [0.0, 0.5, 1.0, 1.5, 2.0]
    // sum = 5.0
    // n = 5
    // lambda = 5 / 5 = 1.0

    let lambda = dist::exponential::estimation::lambda_hat(&data, x_min);

    assert!(
        (lambda - 1.0).abs() < 1e-9,
        "Exponential MLE lambda estimation failed validation."
    );
}
