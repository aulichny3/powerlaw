// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Collection of functions in support of Pareto Type I parameter estimation.

use super::gof;
use crate::stats;
use rayon::prelude::*;

/// Maximum Likelihood Estimator (MLE) for the Pareto Type I alpha parameter.
///
/// This function calculates `alpha_hat = n / sum(ln(x_i / x_min))` for `x_i >= x_min`.
///
/// # Parameters
/// - `data`: A slice of `f64` values representing the dataset.
/// - `x_min`: The minimum value of the distribution.
/// - `sorted`: A boolean indicating whether the `data` slice is already sorted and filtered
///   such that all elements are `>= x_min`. If `false`, the function will filter `data`
///   internally.
///
/// # Returns
/// The estimated alpha parameter as an `f64`. Returns `f64::NAN` if `n` (number of data points >= x_min) is zero.
pub fn type1_alpha_hat(data: &[f64], x_min: f64, sorted: bool) -> f64 {
    // Shared calculation for both branches (using iteration)
    let (n, sum): (f64, f64) = match sorted {
        false => {
            data.iter()
                .filter(|&x| x >= &x_min)
                .map(|&x| (x / x_min).ln()) // Calculate log value
                .fold((0.0, 0.0), |(count, acc_sum), log_val| {
                    (count + 1.0, acc_sum + log_val) // Simultaneously count and sum
                })
        }
        true => {
            // Assuming data is already filtered for x >= x_min when Sorted::True is passed
            data.iter()
                .map(|&x| (x / x_min).ln())
                .fold((0.0, 0.0), |(count, acc_sum), log_val| {
                    (count + 1.0, acc_sum + log_val)
                })
        }
    };

    if n == 0.0 {
        return f64::NAN; // Handle empty dataset case
    }

    n / sum
}

/// Calculates the maximum likelihood estimate (MLE) for the Pareto Type I
/// alpha parameter for a given `x_min` value in a range of minimum values,
/// where each `x_min` is taken from the input data.
///
/// This function is an O(N log N) optimization of the exhaustive search
/// (which is O(N²)). The complexity comes from the initial sort (O(N log N))
/// and the subsequent calculations (O(N)). It achieves this by pre-calculating
/// a suffix sum of the logarithms of the data.
///
/// # Numerical Stability Warning
/// Due to the nature of the optimization, the denominator sum is computed as the
/// difference between two large, nearly equal numbers:
/// `(Sum [from j=i to N-1] of natural_log(data[j])) - ((N - i) * natural_log(data[i]))`.
/// This structure introduces a minor risk of catastrophic cancellation in
/// double-precision floating-point arithmetic (f64), potentially leading
/// to a small loss of precision in the least significant digits. In testing,
/// stability has been held to approximately 12 decimal points.
///
/// # Parameters
/// - `data`: A mutable slice of `f64` values representing the sample dataset.
///   **The data will be sorted in place.**
///
/// # Returns
/// A tuple of two `Vec<f64>`:
/// - `(0)`: A `Vec<f64>` containing the `x_min` values used (which are `data[0]`
///   through `data[N-2]` after sorting).
/// - `(1)`: A `Vec<f64>` containing the corresponding MLE alpha estimates.
pub fn find_alphas_fast(data: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    // 1. Sort the data: O(N log N)
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_total = data.len();
    if n_total <= 1 {
        return (Vec::new(), Vec::new());
    }

    // 2. Pre-calculate ln(x) for all data points: O(N)
    let logs: Vec<f64> = data.iter().map(|&x| x.ln()).collect();

    // 3. Pre-calculate the Suffix Sum of logs: O(N)
    // C[i] = sum(logs[j] for j >= i)
    let mut c_sums: Vec<f64> = vec![0.0; n_total];
    let mut current_sum = 0.0;
    for i in (0..n_total).rev() {
        current_sum += logs[i];
        c_sums[i] = current_sum;
    }

    // 4. Calculate alpha_hats: O(N)
    let mut alphas: Vec<f64> = Vec::with_capacity(n_total - 1);
    let mut x_mins: Vec<f64> = Vec::with_capacity(n_total - 1);

    // Iterate over all but the last value (since data[i..] must have at least one point)
    for i in 0..n_total - 1 {
        let x_min = data[i];
        let n = (n_total - i) as f64; // The count of elements in the slice data[i..]

        // Sum of logs of x_j / x_min:
        // S_i = C_i - (N-i) * ln(x_min)
        let sum_log_ratio = c_sums[i] - n * x_min.ln();

        // Calculate alpha_hat
        // a_hat = n / S_i
        let alpha = n / sum_log_ratio;

        alphas.push(alpha);
        x_mins.push(x_min);
    }

    (x_mins, alphas)
}

/// Generates `n` Pareto Type I alpha estimates for a given `x_min` from
/// the data by repeatedly calling the Pareto Type I alpha MLE function over
/// the dataset. Given it uses the sample data as `x_min`, it may not be
/// precise for small sample sizes.
///
/// This function is slower than [find_alphas_fast] as it uses an iterative approach
/// to calculate the MLE for each `x_min` by looping over the data. However, it may
/// provide higher precision albeit with a performance penalty. It is *not*
/// implemented in this binary, but is available through the public API.
///
/// # Parameters
/// - `data`: A mutable slice of `f64` values representing the sample dataset.
///   **The data will be sorted in place.**
///
/// # Returns
/// A tuple of two `Vec<f64>`:
/// - `(0)`: A `Vec<f64>` containing the `x_min` values used (which are `data[0]`
///   through `data[N-2]` after sorting).
/// - `(1)`: A `Vec<f64>` containing the corresponding MLE alpha estimates.
pub fn find_alphas_exhaustive(data: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    //sort in place to avoid cloning - what if we didn't sort? then we don't need the mutable reference
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut alphas: Vec<f64> = Vec::with_capacity(data.len() - 1);

    // Iterate over the sorted data and compute alpha for each x_min
    for i in 0..data.len() - 1 {
        let x_min = data[i];
        let data_tail = &data[i..]; // Get the slice for the tail

        let alpha = type1_alpha_hat(data_tail, x_min, true);
        //let alpha = dist::powerlaw::alpha_hat(data_tail, x_min);
        alphas.push(alpha);
    }

    // Gemini says this is slow, criterion confirms.
    (data[..data.len() - 1].to_vec(), alphas)
}

/// Estimates the uncertainty (standard deviation) of the `x_min` and `alpha` parameters
/// using a bootstrapping approach with synthetic datasets.
///
/// This function generates `m_simulations` synthetic datasets by uniformly sampling with replacement
/// from the original `data`. For each synthetic dataset, it fits a Pareto Type I
/// distribution to find its best-fit `x_min` and `alpha`. The standard deviations
/// of these `m_simulations` estimated `x_min` and `alpha` values are then returned as the uncertainty estimates.
///
/// This procedure is proposed in Section 3.3 of Clauset, Aaron, et al.
/// [doi:10.48550/ARXIV.0706.1062](https://doi.org/10.48550/arXiv.0706.1062).
///
/// # Parameters
/// - `data`: A slice of `f64` values representing the observed dataset.
/// - `m_simulations`: The number of synthetic datasets to generate for bootstrapping.
///
/// # Returns
/// A tuple `(x_min_std_dev, alpha_std_dev)` where:
/// - `x_min_std_dev`: The standard deviation of the estimated `x_min` values from the simulations.
/// - `alpha_std_dev`: The standard deviation of the estimated `alpha` values from the simulations.
pub fn param_est(data: &[f64], m_simulations: usize) -> (f64, f64) {
    // This function is slow and there has to be a more efficient way of calculating parameter uncertainty.
    let n: usize = data.len();
    let results: Vec<(f64, f64)> = (0..m_simulations)
        .into_par_iter()
        .map(|_| {
            let mut sample = stats::random::random_choice(data, n);
            let a = find_alphas_fast(&mut sample);
            let gof_est = gof(&sample, &a.0, &a.1);

            (gof_est.x_min, gof_est.alpha)
        })
        .collect();

    let (xm, al): (Vec<f64>, Vec<f64>) = results.into_iter().unzip();

    let x_min_est = stats::descriptive::variance(&xm, 1).powf(1. / 2.);
    let alpha_est = stats::descriptive::variance(&al, 1).powf(1. / 2.);

    (x_min_est, alpha_est)
}

/// Represents the log-likelihood of Pareto Type I parameters given a dataset.
#[derive(Debug, Clone)]
pub struct Loglikelihood {
    /// The minimum value of the distribution (x_min).
    pub x_min: f64,
    /// The scaling parameter (alpha) of the distribution.
    pub alpha: f64,
    /// The calculated log-likelihood value.
    pub ll: f64,
}

/// Calculates the log-likelihood `L(theta|x)` of the parameters (`alpha`, `x_min`) given the data.
///
/// This function is currently not used within the main binary, but is available through the public API.
/// It iterates through potential `x_min` and `alpha` pairs and calculates the log-likelihood for each.
///
/// # Parameters
/// - `data`: A mutable slice of `f64` values representing the observed dataset.
///   **The data will be sorted in place.**
/// - `x_mins`: A slice of `f64` values representing the candidate `x_min` values.
/// - `alphas`: A slice of `f64` values representing the candidate `alpha` values,
///   corresponding to the `x_mins`.
///
/// # Returns
/// A `Loglikelihood` struct containing the `x_min`, `alpha`, and `ll` (log-likelihood)
/// for the parameter pair that yielded the maximum log-likelihood.
pub fn likelihood(data: &mut [f64], x_mins: &[f64], alphas: &[f64]) -> Loglikelihood {
    let mut best_ll_result = Loglikelihood {
        x_min: 0.0,
        alpha: 0.0,
        ll: f64::NEG_INFINITY,
    };

    //sort in place to avoid cloning - what if we didn't sort?
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n: usize = data.len();
    // Iterate over the sorted data and compute alpha for each x_min
    for i in 0..n - 1 {
        let x_min = x_mins[i];
        let alpha = alphas[i];
        let data_tail = &data[i..]; // Get the slice for the tail

        // calculate the density of each x given the parameters
        //let ll: f64 = data_tail.iter() // 1. Start with a range iterator
        //.map(|x| dist::pareto::type1_pdf(x_min, alpha, *x).ln()).sum();
        let ll: f64 = data_tail
            .iter()
            .map(|x| (x_min.powf(alpha) * x.powf(-1. - alpha) * alpha).ln())
            .sum();

        if ll > best_ll_result.ll {
            best_ll_result = Loglikelihood {
                x_min: x_min,
                alpha: alpha,
                ll: ll,
            };
        }
    }

    best_ll_result
}
