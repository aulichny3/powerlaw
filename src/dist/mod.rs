// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

/// Contains the exponential distribution implementation.
pub mod exponential;
/// Contains the Pareto Type I distribution implementation and analysis functions.
pub mod pareto;
/// Contains the generic power-law distribution implementation.
pub mod powerlaw;
/// Contains the log-normal distribution implementation.
pub mod lognormal;

pub trait Distribution {
    /// Probability density function (PDF) at a given point x.
    ///
    /// f(x)dx = 1
    fn pdf(&self, x: f64) -> f64;

    /// Cumulative Distribution Function (CDF) value at a given point x.
    ///
    /// F(x) = P(X <= x)
    fn cdf(&self, x: f64) -> f64;

    /// Complementary Cumulative Distribution Function (CCDF) aka survival function, value at a given point x.
    ///
    /// P(X > x) = 1 - F(x)
    fn ccdf(&self, x: f64) -> f64;

    /// Random variate generator for the distribution where parameter u is u(0,1).
    fn rv(&self, u: f64) -> f64;
}
