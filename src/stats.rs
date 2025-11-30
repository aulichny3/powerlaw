// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Module to aid in statistical inference. Contains functions for basic descriptive statistics, non parametric methods for comparing distributions etc.

/// A collection of descriptive statistics, such as mean, variance, etc.
pub mod descriptive {
    /// Calculates the arithmetic mean of a vector.
    ///
    /// # Parameters
    /// - `data`: The input slice of `f64` values.
    ///
    /// # Returns
    /// The arithmetic mean as an `f64`.
    ///
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

    /// Calculates the variance of a vector where `ddof` (degrees of freedom) determines
    /// whether the sample variance (`ddof=1`) or population variance (`ddof=0`) is returned.
    ///
    /// # Parameters
    /// - `data`: The input slice of `f64` values.
    /// - `ddof`: Degrees of freedom. Use `1` for sample variance, `0` for population variance.
    ///
    /// # Returns
    /// The calculated variance as an `f64`.
    ///
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

/// Functions in support of randomization, including sampling and random variate generation.
pub mod random {
    use rand::prelude::*;

    /// Sample `n` elements with replacement from the input `data`.
    ///
    /// # Parameters
    /// - `data`: The input slice of `f64` values to sample from.
    /// - `size`: The number of elements to sample.
    ///
    /// # Returns
    /// A new `Vec<f64>` containing the sampled elements.
    ///
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

    /// Generate `n` random variates from a Uniform(0,1) distribution.
    ///
    /// # Parameters
    /// - `n`: The number of random variates to generate.
    ///
    /// # Returns
    /// A `Vec<f64>` containing `n` random numbers between 0.0 (inclusive) and 1.0 (exclusive).
    pub fn random_uniform(n: usize) -> Vec<f64> {
        let mut rng = rand::rng();

        // Use a range and an iterator chain to generate the vector
        (0..n).map(|_| rng.random_range(0.0..1.0)).collect()
    }
}

/// Supporting functions for Kolmogorov–Smirnov testing for similarity between empirical and reference cumulative distribution functions.
///
/// Given this function is called iteratively over the data during the goodness of fit portion in [crate::dist::pareto::gof()],
/// it requires the observed data to be sorted.
pub mod ks {
    use crate::dist::Distribution;
    /// 1 sample KS test based on a known cdf. This function requires a generic cdf closure/function such as what is defined in the [Distribution] trait.
    ///
    /// # Parameters
    /// - `sorted_x`: The input slice of `f64` values, which must be sorted in ascending order.
    /// - `cdf_func`: A closure or function that computes the theoretical CDF value for a given `f64`.
    ///
    /// # Returns
    /// A tuple `(D_plus, D_minus, D)` where:
    /// - `D_plus`: The D+ statistic.
    /// - `D_minus`: The D- statistic.
    /// - `D`: The overall Kolmogorov-Smirnov statistic, which is `max(D_plus, D_minus)`.
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

    /// The D+ statistic measures the largest amount by which the Empirical Cumulative Distribution Function (ECDF)
    /// is above the theoretical Cumulative Distribution Function (CDF).
    ///
    /// # Parameters
    /// - `cdfvals`: A slice of theoretical CDF values corresponding to the sorted data points.
    /// - `n`: The number of data points.
    ///
    /// # Returns
    /// The D+ statistic as an `f64`.
    fn compute_dplus(cdfvals: &[f64], n: usize) -> f64 {
        (1..=n)
            .map(|i| i as f64 / n as f64 - cdfvals[i - 1])
            .fold(f64::MIN, f64::max)
    }

    /// The D- statistic measures the largest amount by which the theoretical Cumulative Distribution Function (CDF)
    /// is above the Empirical Cumulative Distribution Function (ECDF).
    ///
    /// # Parameters
    /// - `cdfvals`: A slice of theoretical CDF values corresponding to the sorted data points.
    /// - `n`: The number of data points.
    ///
    /// # Returns
    /// The D- statistic as an `f64`.
    fn compute_dminus(cdfvals: &[f64], n: usize) -> f64 {
        (0..n)
            .map(|i| cdfvals[i] - i as f64 / n as f64)
            .fold(f64::MIN, f64::max)
    }
}

pub mod compare {
    use crate::{stats::descriptive, util::erf};

    /// Performs Vuong's closeness test for comparing two non-nested distributions.
    ///
    /// This test determines whether one distribution is significantly closer to the true data
    /// generating process than another. It takes as input the log-likelihoods of the observed
    /// data under each of the two distributions being compared.
    ///
    /// The test calculates a Z-statistic and a corresponding p-value. A significant p-value
    /// (typically p < 0.05) suggests that one distribution is statistically preferred over the other.
    /// The sign of the Z-statistic indicates which distribution is preferred:
    /// - A positive Z-statistic suggests `dist1` is preferred.
    /// - A negative Z-statistic suggests `dist2` is preferred.
    ///
    /// # Parameters
    /// - `dist1`: A slice of log-likelihoods of the observed data under the first distribution.
    /// - `dist2`: A slice of log-likelihoods of the observed data under the second distribution.
    ///
    /// # Returns
    /// A tuple `(Z, p_value)` where:
    /// - `Z`: The Vuong's Z-statistic.
    /// - `p_value`: The two-sided p-value for the test.
    ///
    /// # References
    /// - Vuong, Q. H. (1989). "Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses." Econometrica, 57(2), 307-333. doi:10.2307/1912557
    /// - See also: <https://en.wikipedia.org/wiki/Vuong%27s_closeness_test>
    pub fn vuongs_test(dist1: &[f64], dist2: &[f64]) -> (f64, f64) {

        let m: Vec<f64>  = dist1.iter().zip(dist2.iter()).map(|(a,b)| a - b).collect();
        let mu_m = descriptive::mean(&m);
        let sigma_m = descriptive::variance(&m, 1);
        let n: f64 = dist1.len() as f64;
        
        // Calculate Z score
        let Z: f64 = (mu_m * n.powf(1./2.)) / sigma_m;

        let cdf: f64 = 0.5 * (1. + erf(Z / (2.0 as f64).sqrt()));
        // calculate p-value
        let p_value: f64 = 2. * (1. - cdf);

        (Z, p_value)
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
