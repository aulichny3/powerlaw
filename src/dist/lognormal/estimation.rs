// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Estimation functions for the Lognormal distribution.

use crate::stats::descriptive::{mean, variance};
use crate::util::erf;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Estimates the Maximum Likelihood Estimate (MLE) for the parameters of a Lognormal distribution
/// using a serial (single-threaded) approach.
///
/// This implementation uses the Newton-Raphson method (https://en.wikipedia.org/wiki/Newton%27s_method)
/// to find the roots of the gradient of the log-likelihood function. This is a second-order
/// iterative optimization method that uses both the gradient and the Hessian matrix for faster convergence
/// compared to derivative-free methods like Nelder-Mead.
///
/// This implementation specifically handles data truncated at `x_min`.
pub fn lognormal_mle_truncated_serial(data: &[f64], x_min: f64) -> (f64, f64) {
    let clean_data: Vec<f64> = data.iter().filter(|&&x| x >= x_min).copied().collect();

    if clean_data.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let n = clean_data.len() as f64;
    let log_data: Vec<f64> = clean_data.iter().map(|&x| x.ln()).collect();

    let mu_init = mean(&log_data);
    let sigma_init = variance(&log_data, 0).sqrt();

    let mut mu = mu_init;
    let mut sigma = sigma_init;

    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // Serial summation
        let (sum_diff, sum_sq_diff) = log_data
            .iter()
            .map(|&lx| {
                let diff = lx - mu;
                (diff, diff * diff)
            })
            .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        // Logarithm of the lower bound
        let ln_xmin = x_min.ln();
        // Standardized score of x_min
        let z_min = (ln_xmin - mu) / sigma;
        // PDF of standard normal at z_min
        let phi_zmin = (-0.5 * z_min * z_min).exp() / (2.0 * PI).sqrt();
        // Survival function (1 - CDF) of standard normal at z_min
        let survival_zmin = 0.5 * (1.0 - erf(z_min / 2.0_f64.sqrt()));

        // Inverse Mills ratio (hazard function) at z_min https://en.wikipedia.org/wiki/Mills_ratio
        let lambda_z = if survival_zmin < 1e-15 {
            z_min
        } else {
            phi_zmin / survival_zmin
        };

        // First derivative of Log-Likelihood wrt mu (gradient)
        let dL_dmu = (1.0 / (sigma * sigma)) * sum_diff - (n / sigma) * lambda_z;
        // First derivative of Log-Likelihood wrt sigma (gradient)
        let dL_dsigma =
            -n / sigma + (1.0 / (sigma.powi(3))) * sum_sq_diff - (n / sigma) * z_min * lambda_z;

        // Derivative of the Inverse Mills ratio
        let lambda_prime = lambda_z * lambda_z - z_min * lambda_z;
        // Second derivative wrt mu (Hessian component)
        let d2L_dmu2 = -n / (sigma * sigma) + (n / (sigma * sigma)) * lambda_prime;
        // Second derivative wrt sigma (Hessian component)
        let d2L_dsigma2 = (n / (sigma * sigma)) - (3.0 / sigma.powi(4)) * sum_sq_diff
            + (n / (sigma * sigma)) * (2.0 * z_min * lambda_z + z_min * z_min * lambda_prime);
        // Mixed second derivative (Hessian component)
        let d2L_dmudsigma = (-2.0 / sigma.powi(3)) * sum_diff
            + (n / (sigma * sigma)) * (lambda_z + z_min * lambda_prime);

        // Determinant of the Hessian matrix
        let det = d2L_dmu2 * d2L_dsigma2 - d2L_dmudsigma * d2L_dmudsigma;

        if det.abs() < 1e-12 {
            break;
        }

        // Step size for mu (using Cramer's rule/matrix inversion) https://en.wikipedia.org/wiki/Cramer%27s_rule
        let diff_mu = (d2L_dsigma2 * dL_dmu - d2L_dmudsigma * dL_dsigma) / det;
        // Step size for sigma (using Cramer's rule/matrix inversion)
        let diff_sigma = (-d2L_dmudsigma * dL_dmu + d2L_dmu2 * dL_dsigma) / det;

        // Update parameters
        mu -= diff_mu;
        sigma -= diff_sigma;

        if diff_mu.abs() < tol && diff_sigma.abs() < tol {
            break;
        }
        if sigma <= 0.0 {
            sigma = 1e-5;
        }
    }

    (mu, sigma)
}

/// Estimates the Maximum Likelihood Estimate (MLE) for the parameters of a Lognormal distribution
/// using a parallel (Rayon) approach.
///
/// This implementation uses the Newton-Raphson method (https://en.wikipedia.org/wiki/Newton%27s_method)
/// to find the roots of the gradient of the log-likelihood function. This is a second-order
/// iterative optimization method that uses both the gradient and the Hessian matrix for faster convergence
/// compared to derivative-free methods like Nelder-Mead.
///
/// This implementation specifically handles data truncated at `x_min`.
pub fn lognormal_mle_truncated_par(data: &[f64], x_min: f64) -> (f64, f64) {
    let clean_data: Vec<f64> = data.par_iter().filter(|&&x| x >= x_min).copied().collect();

    if clean_data.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let n = clean_data.len() as f64;
    let log_data: Vec<f64> = clean_data.par_iter().map(|&x| x.ln()).collect();

    let mu_init = mean(&log_data);
    let sigma_init = variance(&log_data, 0).sqrt();

    let mut mu = mu_init;
    let mut sigma = sigma_init;

    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // Parallel summation
        let (sum_diff, sum_sq_diff) = log_data
            .par_iter()
            .map(|&lx| {
                let diff = lx - mu;
                (diff, diff * diff)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        // Logarithm of the lower bound
        let ln_xmin = x_min.ln();
        // Standardized score of x_min
        let z_min = (ln_xmin - mu) / sigma;
        // PDF of standard normal at z_min
        let phi_zmin = (-0.5 * z_min * z_min).exp() / (2.0 * PI).sqrt();
        // Survival function (1 - CDF) of standard normal at z_min
        let survival_zmin = 0.5 * (1.0 - erf(z_min / 2.0_f64.sqrt()));

        // Inverse Mills ratio (hazard function) at z_min
        let lambda_z = if survival_zmin < 1e-15 {
            z_min
        } else {
            phi_zmin / survival_zmin
        };

        // First derivative of Log-Likelihood wrt mu (gradient)
        let dL_dmu = (1.0 / (sigma * sigma)) * sum_diff - (n / sigma) * lambda_z;
        // First derivative of Log-Likelihood wrt sigma (gradient)
        let dL_dsigma =
            -n / sigma + (1.0 / (sigma.powi(3))) * sum_sq_diff - (n / sigma) * z_min * lambda_z;

        // Derivative of the Inverse Mills ratio
        let lambda_prime = lambda_z * lambda_z - z_min * lambda_z;
        // Second derivative wrt mu (Hessian component)
        let d2L_dmu2 = -n / (sigma * sigma) + (n / (sigma * sigma)) * lambda_prime;
        // Second derivative wrt sigma (Hessian component)
        let d2L_dsigma2 = (n / (sigma * sigma)) - (3.0 / sigma.powi(4)) * sum_sq_diff
            + (n / (sigma * sigma)) * (2.0 * z_min * lambda_z + z_min * z_min * lambda_prime);
        // Mixed second derivative (Hessian component)
        let d2L_dmudsigma = (-2.0 / sigma.powi(3)) * sum_diff
            + (n / (sigma * sigma)) * (lambda_z + z_min * lambda_prime);

        // Determinant of the Hessian matrix
        let det = d2L_dmu2 * d2L_dsigma2 - d2L_dmudsigma * d2L_dmudsigma;

        if det.abs() < 1e-12 {
            break;
        }

        // Step size for mu (using Cramer's rule/matrix inversion)
        let diff_mu = (d2L_dsigma2 * dL_dmu - d2L_dmudsigma * dL_dsigma) / det;
        // Step size for sigma (using Cramer's rule/matrix inversion)
        let diff_sigma = (-d2L_dmudsigma * dL_dmu + d2L_dmu2 * dL_dsigma) / det;

        // Update parameters
        mu -= diff_mu;
        sigma -= diff_sigma;

        if diff_mu.abs() < tol && diff_sigma.abs() < tol {
            break;
        }
        if sigma <= 0.0 {
            sigma = 1e-5;
        }
    }

    (mu, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lognormal_mle_truncated() {
        let data = vec![
            1.1051709180756477,
            1.1275010642838382,
            1.157833501704987,
            1.1618342426980598,
            1.185341648052163,
            1.1919864239855598,
            1.2001183141208003,
            1.201650379237617,
            1.2185244589201918,
            1.2223842106725357,
            1.232338006456079,
            1.250550060934094,
            1.2586794697921966,
            1.272108502330761,
            1.284242781489437,
            1.2858852504889506,
            1.300582239401734,
            1.3023259969248246,
            1.328373307564757,
            1.334346857186171,
        ];
        let x_min = 1.1;

        let (mu_s, sigma_s) = lognormal_mle_truncated_serial(&data, x_min);
        let (mu_p, sigma_p) = lognormal_mle_truncated_par(&data, x_min);

        // Data should center around ln(1.22) ~ 0.2 with small spread.
        assert!((mu_s - 0.2).abs() < 0.1);
        assert!((sigma_s - 0.05).abs() < 0.1);

        assert!((mu_s - mu_p).abs() < 1e-9);
        assert!((sigma_s - sigma_p).abs() < 1e-9);
    }
}
