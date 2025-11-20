// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Represents the Pareto Type I density function parameterized by its alpha and minimum x value.
//! P(X=x) = x_min^α * x^(-1 - 𝛼) * α
//!
//! This is computationally equivalent to α * x_min.powf(α) / x.powf(α-1)

pub mod estimation;
pub mod gof;
pub mod hypothesis;

pub use self::estimation::{find_alphas_exhaustive, find_alphas_fast};
pub use self::gof::gof;
pub use self::hypothesis::hypothesis_test;

use crate::dist::Distribution;

pub struct Pareto {
    pub alpha: f64,
    pub x_min: f64,
}

impl Distribution for Pareto {
    /// f(x)dx = 1
    fn pdf(&self, x: f64) -> f64 {
        /*
        The parameterization used is the same as Wolfram Mathematica.
        It is computationally equivalent to:
        α * x_min.powf(α) / x.powf(α-1)
        */
        if x >= self.x_min {
            return self.x_min.powf(self.alpha) * x.powf(-1. - self.alpha) * self.alpha;
        }
        0.
    }

    /// F(x) = P(X <= x)
    fn cdf(&self, x: f64) -> f64 {
        if x >= self.x_min {
            return 1. - (self.x_min / x).powf(self.alpha);
        }
        0.
    }

    /// P(X > x) = 1 - F(x)
    fn ccdf(&self, x: f64) -> f64 {
        /*
        Complimentary cumulative distribution function (CCDF) also known as the survival function.
        */
        1. - self.cdf(x)
    }

    /// Random variate generation via inverse transform where u ~ Uniform(0,1).
    fn rv(&self, u: f64) -> f64 {
        self.x_min / (1. - u).powf(1. / self.alpha)
    }

    /// Calculates the log-likelihood of the data given the distribution.
    fn loglikelihood(&self, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&x| self.pdf(x).ln()).collect()
    }
}

impl Pareto {
    pub fn new(alpha: f64, x_min: f64) -> Self {
        if alpha <= 0. || x_min <= 0. {
            panic!("alpha and x_min parameters for Pareto Type I must be positive. (a > 0, x_min > 0).");
        }
        Pareto { alpha, x_min }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn pareto_params() {
        let _invalid = Pareto::new(-1., -1.);
    }

    #[test]
    fn loglikelihood() {
        let pareto = Pareto::new(2.5, 1.0);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ll = pareto.loglikelihood(&data);
        let expected = vec![
            0.9162907318741551,
            -1.5097244000856536,
            -2.9288522784642286,
            -3.9357395320454622,
            -4.716741961645196,
        ];
        for (a, b) in ll.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}