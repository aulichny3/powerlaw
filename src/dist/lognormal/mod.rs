// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Lognormal distribution parameterized by the mean (mu) and standard deviation (sigma)
//! of the variable's natural logarithm.

use super::Distribution;
use crate::dist::pareto::gof::ParetoFit;
use crate::util::erf;
use rand::Rng;

pub mod estimation;

/// Represents a Lognormal distribution.
///
/// A random variable X is log-normally distributed if its natural logarithm ln(X)
/// is normally distributed.
///
/// # Fields
/// - `mu`: The mean (μ) of the logarithm of the variable.
/// - `sigma`: The standard deviation (σ) of the logarithm of the variable. Must be positive.
/// - `x_min`: The minimum value of the distribution (truncation point). Values below this are prob 0.
pub struct Lognormal {
    pub mu: f64,
    pub sigma: f64,
    pub x_min: f64,
}

impl Distribution for Lognormal {
    /// Calculates the probability density function (PDF) at a given point `x`.
    ///
    /// The PDF for a Lognormal distribution is given by:
    /// f(x; μ, σ) = (1 / (x * σ * sqrt(2π))) * exp(-((ln(x) - μ)^2) / (2σ^2)) for x > 0
    ///
    /// If x_min > 0, the PDF is renormalized: f_trunc(x) = f(x) / (1 - F(x_min)) for x >= x_min.
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x < self.x_min {
            return 0.0;
        }

        let term1 = 1.0 / (x * self.sigma * (2.0 * std::f64::consts::PI).sqrt());
        let term2 = (-(x.ln() - self.mu).powi(2) / (2.0 * self.sigma.powi(2))).exp();
        let raw_pdf = term1 * term2;

        if self.x_min > 0.0 {
            let normalization = self.normalization();
            if normalization == 0.0 {
                return 0.0;
            }
            return raw_pdf / normalization;
        }

        raw_pdf
    }

    /// Calculates the cumulative distribution function (CDF) value at a given point `x`.
    ///
    /// The CDF for a Lognormal distribution is given by:
    /// F(x; μ, σ) = 0.5 * (1 + erf((ln(x) - μ) / (σ * sqrt(2))))
    ///
    /// If x_min > 0, F_trunc(x) = (F(x) - F(x_min)) / (1 - F(x_min)).
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x < self.x_min {
            return 0.0;
        }

        let raw_cdf_x = self.raw_cdf(x);

        if self.x_min > 0.0 {
            let raw_cdf_xmin = self.raw_cdf(self.x_min);
            let normalization = 1.0 - raw_cdf_xmin;
            if normalization == 0.0 {
                return 1.0;
            }
            return (raw_cdf_x - raw_cdf_xmin) / normalization;
        }

        raw_cdf_x
    }

    /// Calculates the complementary cumulative distribution function (CCDF)
    /// (also known as the survival function) value at a given point `x`.
    fn ccdf(&self, x: f64) -> f64 {
        // S_trunc(x) = S(x) / S(x_min)
        if x < self.x_min {
            return 1.0;
        }
        if x <= 0.0 {
            return 1.0;
        } // Should handle x <= 0 if x_min=0

        let raw_ccdf_x = 1.0 - self.raw_cdf(x);

        if self.x_min > 0.0 {
            let raw_ccdf_xmin = 1.0 - self.raw_cdf(self.x_min);
            if raw_ccdf_xmin == 0.0 {
                return 0.0;
            }
            return raw_ccdf_x / raw_ccdf_xmin;
        }

        raw_ccdf_x
    }

    /// Generates a random variate from the Lognormal distribution.
    ///
    /// This implementation uses the Box-Muller transform to generate a standard normal
    /// variate, which is then transformed into a log-normal variate.
    /// It uses rejection sampling to respect x_min.
    ///
    /// # Parameters
    /// - `u`: A random number drawn from a Uniform(0,1) distribution.
    ///
    /// # Returns
    /// A random variate `x` from the Lognormal distribution.
    fn rv(&self, u1: f64) -> f64 {
        let mut rng = rand::rng();
        let mut current_u1 = u1;

        // Safety break
        let mut tries = 0;
        let max_tries = 10000;

        loop {
            let u2: f64 = rng.random();

            // Box-Muller transform to get a standard normal variate
            let z = (-2.0 * current_u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            // Transform the standard normal variate to a log-normal variate
            let x = (self.mu + self.sigma * z).exp();

            if x >= self.x_min {
                return x;
            }

            tries += 1;
            if tries > max_tries {
                // Fallback to x_min to stay in bounds if rejection fails too many times
                // This typically implies x_min is very far in the tail
                return self.x_min;
            }

            // New random u1 for next iteration
            current_u1 = rng.random();
        }
    }

    /// Calculates the log-likelihood of the data given the distribution.
    fn loglikelihood(&self, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&x| self.pdf(x).ln()).collect()
    }

    fn name(&self) -> &'static str {
        "Lognormal"
    }

    fn parameters(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("mu", self.mu),
            ("sigma", self.sigma),
            ("x_min", self.x_min),
        ]
    }
}

impl Lognormal {
    /// Creates a new Lognormal distribution (untruncated, x_min = 0).
    ///
    /// # Parameters
    /// - `mu`: The mean (μ) of the logarithm of the variable.
    /// - `sigma`: The standard deviation (σ) of the logarithm of the variable. Must be positive.
    pub fn new(mu: f64, sigma: f64) -> Self {
        if sigma <= 0.0 {
            panic!("sigma parameter for Lognormal must be positive (σ > 0).");
        }
        Lognormal {
            mu,
            sigma,
            x_min: 0.0,
        }
    }

    /// Creates a new Truncated Lognormal distribution.
    ///
    /// # Parameters
    /// - `mu`: The mean (μ) of the logarithm of the variable.
    /// - `sigma`: The standard deviation (σ) of the logarithm of the variable. Must be positive.
    /// - `x_min`: The minimum value (lower truncation point).
    pub fn new_truncated(mu: f64, sigma: f64, x_min: f64) -> Self {
        if sigma <= 0.0 {
            panic!("sigma parameter for Lognormal must be positive (σ > 0).");
        }
        Lognormal { mu, sigma, x_min }
    }

    /// Creates a new Lognormal distribution by fitting it to data
    /// using the x_min from a previous Pareto fit.
    pub fn from_fitment(data: &[f64], fitment: &ParetoFit) -> Self {
        let x_min = fitment.x_min;
        // Use the truncated MLE estimator
        let (mu, sigma) = estimation::lognormal_mle_truncated_serial(data, x_min);
        Self { mu, sigma, x_min }
    }

    // Helper for raw untruncated CDF
    fn raw_cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        0.5 * (1.0 + erf((x.ln() - self.mu) / (self.sigma * (2.0_f64).sqrt())))
    }

    fn normalization(&self) -> f64 {
        1.0 - self.raw_cdf(self.x_min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::f64;

    #[test]
    fn test_lognormal_pdf() {
        let dist = Lognormal::new(0.0, 1.0);
        // For standard lognormal (mu=0, sigma=1), pdf at x=1 is 1/sqrt(2pi)
        let expected = 1.0 / (2.0 * f64::consts::PI).sqrt();
        assert!((dist.pdf(1.0) - expected).abs() < 1e-7);
        assert_eq!(dist.pdf(0.0), 0.0);
        assert_eq!(dist.pdf(-1.0), 0.0);
    }

    #[test]
    fn test_truncated_pdf() {
        // mu=0, sigma=1, x_min=1.0
        let dist = Lognormal::new_truncated(0.0, 1.0, 1.0);
        // PDF at x < 1.0 should be 0
        assert_eq!(dist.pdf(0.5), 0.0);

        // Normalization: 1 - CDF(1.0) = 1 - 0.5 = 0.5
        // PDF at x=1.0: raw_pdf(1.0) / 0.5
        // raw_pdf(1.0) = 1/sqrt(2pi) ~ 0.3989
        // pdf(1.0) ~ 0.7978
        let expected_raw = 1.0 / (2.0 * f64::consts::PI).sqrt();
        let expected = expected_raw / 0.5;
        assert!((dist.pdf(1.0) - expected).abs() < 1e-7);
    }

    #[test]
    fn test_lognormal_cdf() {
        let dist = Lognormal::new(0.0, 1.0);
        // For standard lognormal (mu=0, sigma=1), cdf at x=1 is 0.5
        assert!((dist.cdf(1.0) - 0.5).abs() < 1e-7);
        // cdf at x=exp(1)
        let expected_exp1 = 0.5 * (1.0 + erf(1.0 / (2.0_f64).sqrt()));
        assert!((dist.cdf(f64::consts::E) - expected_exp1).abs() < 1e-7);
        assert_eq!(dist.cdf(0.0), 0.0);
    }

    #[test]
    fn test_lognormal_ccdf() {
        let dist = Lognormal::new(0.0, 1.0);
        assert!((dist.ccdf(1.0) - (1.0 - 0.5)).abs() < 1e-7);
        let expected_exp1 = 1.0 - (0.5 * (1.0 + erf(1.0 / (2.0_f64).sqrt())));
        assert!((dist.ccdf(f64::consts::E) - expected_exp1).abs() < 1e-7);
        assert_eq!(dist.ccdf(0.0), 1.0);
    }

    #[test]
    fn test_lognormal_rv_generation() {
        let dist = Lognormal::new(0.0, 1.0);
        let mut rng = rand::rng();
        // Check if generated values are positive
        for _ in 0..100 {
            let u: f64 = rng.gen_range(0.001..0.999);
            let val = dist.rv(u);
            assert!(val > 0.0, "Generated value was not positive: {}", val);
        }
    }

    #[test]
    #[should_panic]
    fn test_new_invalid_sigma() {
        Lognormal::new(0.0, -1.0);
    }

    #[test]
    fn loglikelihood() {
        let dist = Lognormal::new(0.0, 1.0);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ll = dist.loglikelihood(&data);
        let expected = vec![
            -0.9189385332046727,
            -1.8523122207237188,
            -2.6210253022790733,
            -3.266138922160966,
            -3.8235216426288905,
        ];
        for (a, b) in ll.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
