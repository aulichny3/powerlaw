pub mod estimation;
pub use self::estimation::lambda_hat;

// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

/// Exponential distribution parameterized by its lambda parameter.
use super::Distribution;
use crate::dist::pareto::gof::ParetoFit;

/// Represents an Exponential distribution.
///
/// The Shifted Exponential distribution is a continuous probability distribution that describes
/// the time between events in a Poisson point process, i.e., a process in which events
/// occur continuously and independently at a constant average rate. This implementation
/// allows for shifting the distribution s.t Pr(X=x) s.t x_min <= x
///
/// # Fields
/// - `lambda`: The rate parameter (λ) of the distribution. Must be positive.
/// - `x_min`: The minimum value of the distribution (x_m). Must be positive.
pub struct Exponential {
    pub lambda: f64,
    pub x_min: f64,
}

impl Exponential {
    /// Creates a new Exponential distribution by fitting it to data
    /// using the x_min from a previous Pareto fit.
    pub fn from_fitment(data: &[f64], fitment: &ParetoFit) -> Self {
        let x_min = fitment.x_min;
        let lambda = estimation::lambda_hat(&data, x_min);
        Self { lambda, x_min }
    }
}

/// Implements the `Distribution` trait for the `Exponential` distribution.
impl Distribution for Exponential {
    /// Calculates the probability density function (PDF) at a given point `x`.
    ///
    /// The PDF for an Exponential distribution is given by:
    /// f(x; λ, x_min) = λ * e^(-λ * (x-x_min) for x >= 0, and 0 otherwise.
    ///
    /// # Parameters
    /// - `x`: The point at which to evaluate the PDF.
    ///
    /// # Returns
    /// The PDF value at `x`.
    fn pdf(&self, x: f64) -> f64 {
        if x >= self.x_min {
            return self.lambda * std::f64::consts::E.powf(-self.lambda * (x - self.x_min));
        }
        0.
    }

    /// Calculates the cumulative distribution function (CDF) value at a given point `x`.
    ///
    /// The CDF for an Exponential distribution is given by:
    /// F(x; λ, x_min) = 1 - e^(-λ * (x-x_min)) for x >= 0, and 0 otherwise.
    ///
    /// # Parameters
    /// - `x`: The point at which to evaluate the CDF.
    ///
    /// # Returns
    /// The CDF value at `x`.
    fn cdf(&self, x: f64) -> f64 {
        if x >= self.x_min {
            return 1. - std::f64::consts::E.powf(-self.lambda * (x - self.x_min));
        }
        0.
    }

    /// Calculates the complementary cumulative distribution function (CCDF)
    /// (also known as the survival function) value at a given point `x`.
    ///
    /// The CCDF for an Exponential distribution is given by:
    /// P(X > x; λ, x_min) = e^(-λ (x-x_min)) for x >= 0, and 1 otherwise.
    ///
    /// # Parameters
    /// - `x`: The point at which to evaluate the CCDF.
    ///
    /// # Returns
    /// The CCDF value at `x`.
    fn ccdf(&self, x: f64) -> f64 {
        // This can be simplified to `std::f64::consts::E.powf(-self.lambda * x)`
        // but keeping it as `1. - self.cdf(x)` for consistency with the original structure
        // and to avoid functional changes.
        if x >= self.x_min {
            return 1. - self.cdf(x);
        }
        0.
    }

    /// Generates a random variate from the Exponential distribution using the inverse transform method.
    ///
    /// # Parameters
    /// - `u`: A random number drawn from a Uniform(0,1) distribution.
    ///
    /// # Returns
    /// A random variate `x` from the Exponential distribution.
    fn rv(&self, u: f64) -> f64 {
        self.x_min - (1. / self.lambda) * (1. - u).ln() // technically 1-U which is equivalent to U in this context.
    }

    /// Calculates the log-likelihood of the data given the distribution.
    fn loglikelihood(&self, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&x| self.pdf(x).ln()).collect()
    }

    fn name(&self) -> &'static str {
        "Exponential"
    }

    fn parameters(&self) -> Vec<(&'static str, f64)> {
        vec![("lambda", self.lambda), ("x_min", self.x_min)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Distribution;

    #[test]
    fn random_variate() {
        let expo = Exponential {
            lambda: 68.0,
            x_min: 0.0282,
        };

        // X = x
        let x: f64 = expo.rv(0.2);

        assert_eq!(x, 0.03148152281344426);
    }

    #[test]
    fn loglikelihood() {
        let expo = Exponential {
            lambda: 68.0,
            x_min: 0.0282,
        };
        let data = vec![0.03, 0.05, 0.1, 0.15, 0.2];
        let ll = expo.loglikelihood(&data);
        let expected = vec![
            4.097107705176107,
            2.737107705176107,
            -0.6628922948238936,
            -4.062892294823892,
            -7.462892294823894,
        ];
        dbg!(&ll);
        for (a, b) in ll.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn pdf() {
        let expo = Exponential {
            lambda: 2.5,
            x_min: 1.0,
        };
        let x = 2.0;
        let p = expo.pdf(x);
        assert!((p - 0.20521249655).abs() < 1e-9);
    }

    #[test]
    fn cdf() {
        let expo = Exponential {
            lambda: 2.5,
            x_min: 1.0,
        };
        let x = 2.0;
        let c = expo.cdf(x);
        assert!((c - 0.91791500138).abs() < 1e-9);
    }

    #[test]
    fn ccdf() {
        let expo = Exponential {
            lambda: 2.5,
            x_min: 1.0,
        };
        let x = 2.0;
        let c = expo.ccdf(x);
        assert!((c - 0.08208499862).abs() < 1e-9);
    }
}
