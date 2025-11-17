// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Exponential distribution parameterized by its lambda parameter.

use super::Distribution;

/// Represents an Exponential distribution.
///
/// The Exponential distribution is a continuous probability distribution that describes
/// the time between events in a Poisson point process, i.e., a process in which events
/// occur continuously and independently at a constant average rate.
///
/// # Fields
/// - `lambda`: The rate parameter (λ) of the distribution. Must be positive.
pub struct Exponential {
    pub lambda: f64,
}

/// Implements the `Distribution` trait for the `Exponential` distribution.
impl Distribution for Exponential {
    /// Calculates the probability density function (PDF) at a given point `x`.
    ///
    /// The PDF for an Exponential distribution is given by:
    /// f(x; λ) = λ * e^(-λx) for x >= 0, and 0 otherwise.
    ///
    /// # Parameters
    /// - `x`: The point at which to evaluate the PDF.
    ///
    /// # Returns
    /// The PDF value at `x`.
    fn pdf(&self, x: f64) -> f64 {
        self.lambda * std::f64::consts::E.powf(-self.lambda * x)
    }

    /// Calculates the cumulative distribution function (CDF) value at a given point `x`.
    ///
    /// The CDF for an Exponential distribution is given by:
    /// F(x; λ) = 1 - e^(-λx) for x >= 0, and 0 otherwise.
    ///
    /// # Parameters
    /// - `x`: The point at which to evaluate the CDF.
    ///
    /// # Returns
    /// The CDF value at `x`.
    fn cdf(&self, x: f64) -> f64 {
        1. - std::f64::consts::E.powf(-self.lambda * x)
    }

    /// Calculates the complementary cumulative distribution function (CCDF)
    /// (also known as the survival function) value at a given point `x`.
    ///
    /// The CCDF for an Exponential distribution is given by:
    /// P(X > x; λ) = e^(-λx) for x >= 0, and 1 otherwise.
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
        1. - self.cdf(x)
    }

    /// Generates a random variate from the Exponential distribution using the inverse transform method.
    ///
    /// # Parameters
    /// - `u`: A random number drawn from a Uniform(0,1) distribution.
    ///
    /// # Returns
    /// A random variate `x` from the Exponential distribution.
    fn rv(&self, u: f64) -> f64 {
        -u.ln() / self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Distribution;

    #[test]
    fn random_variate() {
        let expo = Exponential { lambda: 4.0 };

        // X = x
        let x: f64 = expo.rv(0.2);

        assert_eq!(x, 0.40235947810852507);
    }
}
