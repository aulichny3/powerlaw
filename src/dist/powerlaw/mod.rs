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
use crate::dist::pareto::gof::ParetoFit;

pub mod estimation;
pub use self::estimation::alpha_hat;

/// Represents a generic Power-Law distribution, which is a continuous, unbounded distribution.
///
/// The probability density function (PDF) is defined as:
/// f(x) = C * x^(-alpha)
/// where C is the normalization constant: (alpha - 1) * x_min^(alpha - 1)
/// and x >= x_min.
///
/// This simplifies to a Pareto Type I distribution.
/// Note: The `alpha` parameter here is the power-law exponent. It is equal to `1 + alpha_pareto`,
/// where `alpha_pareto` is the shape parameter of the standard Pareto Type I distribution.
///
/// # Fields
/// - `alpha`: The power-law exponent (α). Must be greater than 1.
/// - `x_min`: The minimum value of the distribution (x_m). Must be positive.
pub struct Powerlaw {
    pub alpha: f64,
    pub x_min: f64,
}

/// Implements the `Distribution` trait for the `Powerlaw` distribution.
impl Distribution for Powerlaw {
    /// Calculates the probability density function (PDF) at a given point `x`.
    ///
    /// f(x) = ((alpha - 1) / x_min) * (x / x_min)^(-alpha)
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
    /// F(x) = 1 - (x / x_min)^(-alpha + 1)
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
    /// S(x) = (x / x_min)^(-alpha + 1)
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

impl Powerlaw {
    pub fn new(alpha: f64, x_min: f64) -> Self {
        if alpha <= 0. || x_min <= 0. {
            panic!("alpha and x_min parameters for powerlaw distribution must be positive. (a > 0, x_min > 0).");
        }
        Powerlaw { alpha, x_min }
    }
}

/// Creates a `Powerlaw` distribution directly from a `ParetoFit` result.
///
/// This converts the Pareto shape parameter to the Powerlaw exponent:
/// `powerlaw_alpha = pareto_fit.alpha + 1.0`
impl From<ParetoFit> for Powerlaw {
    fn from(fitment: ParetoFit) -> Self {
        Self {
            alpha: fitment.alpha + 1.,
            x_min: fitment.x_min,
        }
    }
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

    #[test]
    fn test_alpha_hat() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x_min = 1.0;
        // alpha_hat = 1 + n / sum(ln(x/x_min))
        // sum_ln = 15.104412573
        // n = 10
        // alpha = 1 + 10 / 15.1044... = 1.66205...
        let alpha = alpha_hat(&data, x_min);
        assert!((alpha - 1.66205).abs() < 1e-4);
    }

    #[test]
    fn test_from_pareto_fit() {
        let fit = ParetoFit {
            x_min: 2.5,
            alpha: 1.5,
            d: 0.1,
            len_tail: 100,
        };
        let pl = Powerlaw::from(fit);
        assert_eq!(pl.x_min, 2.5);
        assert_eq!(pl.alpha, 2.5); // 1.5 + 1.0
    }
}
