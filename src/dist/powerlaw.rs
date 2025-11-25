// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! The generic Power-Law distribution defined by Cf(x) where f(x) = x^(-alpha) and C is the normalizing constant to ensure the distribution integrates to 1
//! where C = (alpha - 1) * x_min^(alpha - 1).
//! Continuous, Unbounded Power Law which simplifies to a Pareto Type I.
//! However, the alpha parameter will be exactly 1.0 greater than that of more
//! common expressions such as the Pareto Type I pdf listed at:
//! [https://en.wikipedia.org/wiki/Pareto_distribution](https://en.wikipedia.org/wiki/Pareto_distribution)
use super::Distribution;

/// Represents a generic Power-Law distribution, which is a continuous, unbounded distribution
/// that simplifies to a Pareto Type I distribution.
///
/// # Fields
/// - `alpha`: The scaling parameter (α) of the distribution. Must be greater than 1.
/// - `x_min`: The minimum value of the distribution (x_m). Must be positive.
pub struct Powerlaw {
    pub alpha: f64,
    pub x_min: f64,
}

/// Implements the `Distribution` trait for the `Powerlaw` distribution.
impl Distribution for Powerlaw {
    /// Calculates the probability density function (PDF) at a given point `x`.
    ///
    /// The alpha parameter will be exactly 1.0 greater than that of more
    /// common expressions such as the PDF listed at:
    /// <https://en.wikipedia.org/wiki/Pareto_distribution>
    ///
    /// # Parameters
    /// - `x`: The point at which to evaluate the PDF. Must be greater than or equal to `x_min`.
    ///
    /// # Returns
    /// The PDF value at `x`.
    fn pdf(&self, x: f64) -> f64 {
        (self.alpha - 1.) / self.x_min.powf(1. - self.alpha) * x.powf(-self.alpha)
    }

    /// Calculates the cumulative distribution function (CDF) value at a given point `x`.
    ///
    /// # Parameters
    /// - `x`: The point at which to evaluate the CDF.
    ///
    /// # Returns
    /// The CDF value at `x`.
    fn cdf(&self, x: f64) -> f64 {
        1. - (x / self.x_min).powf(-self.alpha + 1.)
    }

    /// Calculates the complementary cumulative distribution function (CCDF)
    /// (also known as the survival function) value at a given point `x`.
    ///
    /// # Parameters
    /// - `x`: The point at which to evaluate the CCDF.
    ///
    /// # Returns
    /// The CCDF value at `x`.
    fn ccdf(&self, x: f64) -> f64 {
        1. - self.cdf(x)
    }

    /// Generates a random variate from the Power-Law distribution using the inverse transform method.
    ///
    /// # Parameters
    /// - `u`: A random number drawn from a Uniform(0,1) distribution.
    ///
    /// # Returns
    /// A random variate `x` from the Power-Law distribution.
    fn rv(&self, u: f64) -> f64 {
        self.x_min * (1. - u).powf(-1. / (self.alpha - 1.))
    }

    /// Calculates the log-likelihood of the data given the distribution.
    fn loglikelihood(&self, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&x| self.pdf(x).ln()).collect()
    }

    fn name(&self) -> &'static str {
        "Powerlaw"
    }

    fn parameters(&self) -> Vec<(&'static str, f64)> {
        vec![("alpha", self.alpha), ("x_min", self.x_min)]
    }
}

/// Calculates the Maximum Likelihood Estimate (MLE) for the alpha parameter of a power-law distribution.
///
/// # Parameters
/// - `data`: A slice of `f64` values representing the data points. It's assumed that all `x` in `data` are `>= x_min`.
/// - `x_min`: The minimum value of the distribution.
///
/// # Returns
/// The estimated alpha parameter as an `f64`.
pub fn alpha_hat(data: &[f64], x_min: f64) -> f64 {
    let n: usize = data.len();

    let logs = data.iter().map(|x: &f64| (x / x_min).ln());
    let sum_of_logs: f64 = logs.sum();

    1. + n as f64 / sum_of_logs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Distribution;

    #[test]
    fn loglikelihood() {
        let pl = Powerlaw {
            alpha: 2.5,
            x_min: 1.0,
        };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ll = pl.loglikelihood(&data);
        let expected = vec![
            0.4054651081081644,
            -1.327402843274342,
            -2.341065613606193,
            -3.060270794686873,
            -3.618129673005666,
        ];
        for (a, b) in ll.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }
}