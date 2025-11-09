// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Module to aid in statistical inference. Contains functions for basic descriptive statistics, non parametric methods for comparing distributions etc.

/// A collection of descriptive statistics, mean, variance etc.
pub mod descriptive {
    /// Calculates the arithmetic mean of a vector.
    /// # Example
    /// ```
    /// use powerlaw::stats;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mu = stats::descriptive::mean(&data); // results in 3.0
    /// ```
    pub fn mean(data: &[f64]) -> f64 {
        let sum: f64 = data.iter().sum();

        sum / data.len() as f64
    }

    /// Calculates the variance of a vector where ddof = degrees of freedom. If ddof=1, the sample variance is returned otherwise the population variance is returned.
    /// # Example
    /// ```
    /// use powerlaw::stats;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let sigma_squared_pop = stats::descriptive::variance(&data, 0); // 2.0
    /// let sigma_squared_samp = stats::descriptive::variance(&data, 1); // 2.5
    /// ```
    pub fn variance(data: &[f64], ddof: u8) -> f64 {
        // calculate mean
        let mu = mean(&data);

        // find squared differences
        let sum_squared_diff: f64 = (0..data.len()).map(|x| (data[x] - mu).powf(2.)).sum();

        // if ddof = 1 then return the sample variance otherwise return the population variance
        if ddof == 1 {
            return sum_squared_diff / (data.len() - 1) as f64;
        }

        sum_squared_diff / data.len() as f64
    }
}

/// Functions in support of randomization.
pub mod random {
    use rand::prelude::*;

    /// Sample *n* elements with probability U(0,1) with replacement.
    /// # Example
    /// ```
    /// use powerlaw::stats;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let X = stats::random::random_choice(&data, 10); // could look like: [2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 3.0, 5.0, 4.0]
    /// ```
    pub fn random_choice(data: &[f64], size: usize) -> Vec<f64> {

        let mut samp: Vec<f64> = vec![];

        // Get a random number generator
        let mut rng = rand::rng();

        for _ in 0..size {
            // Generate a random index in the range [0, vector_length)
            let random_index = rng.random_range(0..data.len());

            // Access the element at the random index
            samp.push(data[random_index]);
        }
        samp
    }

    /// Generate *n* random variates from U(0,1).
    pub fn random_uniform(n: usize) -> Vec<f64> {
        let mut rng = rand::rng();

        // Use a range and an iterator chain to generate the vector
        (0..n).map(|_| rng.random_range(0.0..1.0)).collect()
    }
}

/// Supporting functions for Kolmogorov–Smirnov testing for similarity between empirical and reference cumulative distribution functions.
/// Given this function is called iteratively over the data during the goodness of fit portion in [crate::dist::pareto::gof()], it requires the observed data to be sorted.
pub mod ks {
    use crate::dist::Distribution;
    /// 1 sample KS test based on a known cdf. This function requires a generic cdf closure/function such as what is defined in the [Distribution] trait.
    pub fn ks_1sam_sorted<F>(sorted_x: &[f64], cdf_func: F) -> (f64, f64, f64)
    where
        F: Fn(f64) -> f64, // 'F' is a generic type that must be a closure/function
    {
        let n = sorted_x.len();
        if n == 0 {
            return (0.0, 0.0, 0.0);
        }

        // calculate the theoretical CDF F(x_i)
        let mut cdfvals: Vec<f64> = Vec::with_capacity(n);
        for x in sorted_x.iter() {
            // Use the generic cdf_func closure
            cdfvals.push(cdf_func(*x));
        }

        let dplus = compute_dplus(&cdfvals, n);
        let dminus = compute_dminus(&cdfvals, n);

        let d = dplus.max(dminus);
        (dplus, dminus, d)
    }

    /// The D+ statistic measures the largest amount by which the ECDF is above the theoretical CDF.
    fn compute_dplus(cdfvals: &[f64], n: usize) -> f64 {
        (1..=n)
            .map(|i| i as f64 / n as f64 - cdfvals[i - 1])
            .fold(f64::MIN, f64::max)
    }

    /// The D- statistic measures the largest amount by which the ECDF is below the theoretical CDF.
    fn compute_dminus(cdfvals: &[f64], n: usize) -> f64 {
        /*
        Computes D- as used in the Kolmogorov-Smirnov test.
        ...
        */
        (0..n)
            .map(|i| cdfvals[i] - i as f64 / n as f64)
            .fold(f64::MIN, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use crate::stats::{self, ks::ks_1sam_sorted, random::random_choice};

    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(descriptive::mean(&data), 3.0);
    }

    #[test]
    fn test_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(descriptive::variance(&data, 0), 2.0); // Population variance
        assert_eq!(descriptive::variance(&data, 1), 2.5); // Sample variance
    }

    #[test]
    fn test_random_choice() {
        let X = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let len = random_choice(&X, 3).len();
        assert_eq!(len, 3);
    }

    #[test]
    fn test_ks() {
        // sample some random U(0,1)
        let mut X = stats::random::random_uniform(1000);
        // sort
        X.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // cdf of U(0,1) is the identity function f(x) = x
        let F_x = |x: f64| x;

        let (_, _, d) = ks_1sam_sorted(&X, F_x);

        assert!(d < 0.2, "KS stat for this test should be very small (approach 0) given the empirical is drawn from the theoretical CDF. found KS statistic = {d}");
    }
}
