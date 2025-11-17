// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Lognormal distribution parameterized by the mean (mu) and standard deviation (sigma)
//! of the variable's natural logarithm.

use crate::util::erf;
use super::Distribution;
use rand::Rng;

/// Represents a Lognormal distribution.
///
/// A random variable X is log-normally distributed if its natural logarithm ln(X)
/// is normally distributed.
///
/// # Fields
/// - `mu`: The mean (μ) of the logarithm of the variable.
/// - `sigma`: The standard deviation (σ) of the logarithm of the variable. Must be positive.
pub struct Lognormal {
    pub mu: f64,
    pub sigma: f64,
}

impl Distribution for Lognormal {
    /// Calculates the probability density function (PDF) at a given point `x`.
    ///
    /// The PDF for a Lognormal distribution is given by:
    /// f(x; μ, σ) = (1 / (x * σ * sqrt(2π))) * exp(-((ln(x) - μ)^2) / (2σ^2)) for x > 0
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let term1 = 1.0 / (x * self.sigma * (2.0 * std::f64::consts::PI).sqrt());
        let term2 = (-(x.ln() - self.mu).powi(2) / (2.0 * self.sigma.powi(2))).exp();
        term1 * term2
    }

    /// Calculates the cumulative distribution function (CDF) value at a given point `x`.
    ///
    /// The CDF for a Lognormal distribution is given by:
    /// F(x; μ, σ) = 0.5 * (1 + erf((ln(x) - μ) / (σ * sqrt(2))))
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        0.5 * (1.0 + erf((x.ln() - self.mu) / (self.sigma * (2.0_f64).sqrt())))
    }

    /// Calculates the complementary cumulative distribution function (CCDF)
    /// (also known as the survival function) value at a given point `x`.
    fn ccdf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    /// Generates a random variate from the Lognormal distribution.
    ///
    /// This implementation uses the Box-Muller transform to generate a standard normal
    /// variate, which is then transformed into a log-normal variate.
    /// It uses the provided `u` as one of the two required uniform random numbers
    /// and generates the second one internally. This is a slight misuse of the trait's
    /// intended `rv` functionality but is necessary given the function signature.
    ///
    /// # Parameters
    /// - `u`: A random number drawn from a Uniform(0,1) distribution.
    ///
    /// # Returns
    /// A random variate `x` from the Lognormal distribution.
    fn rv(&self, u1: f64) -> f64 {
        let mut rng = rand::rng();
        let u2: f64 = rng.random();

        // Box-Muller transform to get a standard normal variate
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // Transform the standard normal variate to a log-normal variate
        (self.mu + self.sigma * z).exp()
    }
}

impl Lognormal {
    /// Creates a new Lognormal distribution.
    ///
    /// # Parameters
    /// - `mu`: The mean (μ) of the logarithm of the variable.
    /// - `sigma`: The standard deviation (σ) of the logarithm of the variable. Must be positive.
    pub fn new(mu: f64, sigma: f64) -> Self {
        if sigma <= 0.0 {
            panic!("sigma parameter for Lognormal must be positive (σ > 0).");
        }
        Lognormal { mu, sigma }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;
    use rand::Rng;

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
            let u: f64 = rng.random_range(0.001..0.999);
            let val = dist.rv(u);
            assert!(val > 0.0, "Generated value was not positive: {}", val);
        }
    }

    #[test]
    #[should_panic]
    fn test_new_invalid_sigma() {
        Lognormal::new(0.0, -1.0);
    }
}
