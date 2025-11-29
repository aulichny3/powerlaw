// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Repeatedly KS test a series of x_min and alpha parameter sets and
//! the sample data to find the pair that has the best KS statistic. This is the
//! method proposed in Section 3.3 of
//! Clauset, Aaron and Shalizi, Cosma Rohilla and Newman, M. E. J. [doi:10.48550/ARXIV.0706.1062](https://doi.org/10.48550/arXiv.0706.1062)
//!
use super::Distribution;
use super::Pareto;
use crate::stats;
use rayon::prelude::*;

/// Represents the best-fit parameters and goodness-of-fit statistic for a dataset.
#[derive(Debug, Clone)]
pub struct ParetoFit {
    /// The `x_min` parameter that yields the best fit.
    pub x_min: f64,
    /// The `alpha` parameter that yields the best fit.
    pub alpha: f64,
    /// The Kolmogorov-Smirnov (KS) test statistic `d` for the best fit.
    pub d: f64,
    /// The number of elements in the tail of the distribution (`>= x_min`).
    pub len_tail: usize,
}

/// Performs a goodness-of-fit (GOF) test by iterating over all specified `x_min` and `alpha` pairs
/// to identify the set that has the smallest Kolmogorov-Smirnov (KS) test statistic `D`.
///
/// # Parameters
/// - `data`: A slice of `f64` values representing the sample dataset to fit.
/// - `x_mins`: A slice of `f64` values representing the candidate `x_min` values.
/// - `alphas`: A slice of `f64` values representing the candidate `alpha` values,
///   corresponding to the `x_mins`.
///
/// # Returns
/// A `ParetoFit` struct containing the `x_min`, `alpha`, `d` (KS statistic), and `len_tail`
/// for the parameter pair that best fits the data (i.e., minimizes `d`).
pub fn gof(data: &[f64], x_mins: &[f64], alphas: &[f64]) -> ParetoFit {
    // 1. Sort the data once. O(D log D).
    let mut sorted_data_vec = data.to_vec();
    sorted_data_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // 2. Share the sorted data by reference for all parallel tasks.
    let current_data_slice = sorted_data_vec.as_slice();

    // 3. Parallelize the iteration over x_mins/alphas using Rayon.
    let best_fit = x_mins
        .par_iter()
        .zip(alphas.par_iter()) // Iterate in parallel over both parameter vectors
        .map(|(&x_min, &alpha)| {
            // 4. Find the starting index using binary search (O(log D)).
            // partition_point finds the first index where the predicate is FALSE (i.e., x >= x_min).
            let start_idx = current_data_slice.partition_point(|&x| x < x_min);

            // 5. Get the slice of the tail data *without copying*.
            let thread_data_slice = &current_data_slice[start_idx..];

            // 6. Perform the KS test.
            let custom_cdf = |x: f64| Pareto { x_min, alpha }.cdf(x);

            let ks_result = stats::ks::ks_1sam_sorted(thread_data_slice, custom_cdf);

            ParetoFit {
                x_min,
                alpha,
                d: ks_result.2,
                len_tail: thread_data_slice.len(),
            }
        })
        .min_by(|a, b| a.d.partial_cmp(&b.d).unwrap()) // Find the best fit in parallel
        .unwrap(); // Unwrap is safe if x_mins/alphas are non-empty

    // The statistical error for the alpha parameter is reported in wikipedia as having this calculation based on:
    // MEJ Newman (2005) Power laws, Pareto distributions and Zipf's law, Contemporary Physics, 46:5, 323-351, DOI: 10.1080/00107510500052444
    // However a lit review of the cited material indicated no such thing.
    // best_fit.alpha_SE = best_fit.alpha / (best_fit.len_tail as f64).sqrt();

    best_fit.clone()
}
