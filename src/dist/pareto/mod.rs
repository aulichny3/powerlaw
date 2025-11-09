// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Represents the Pareto Type I density function parameterized by its alpha and minimum x value.
//! P(X=x) = x_min^alpha * x^(-1 - alpha) * alpha
//!
//! This is computationally equivalent to alpha * x_min.powf(alpha) / x.powf(alpha-1)cargo cle

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
    fn pdf(&self, x: f64) -> f64 {
        /*
        The parameterization used is the same as Wolfram Mathematica.
        It is computationally equivalent to:
        alpha * x_min.powf(alpha) / x.powf(alpha-1)
        */
        //! f(x)dx = 1
        if x >= self.x_min {
            return self.x_min.powf(self.alpha) * x.powf(-1. - self.alpha) * self.alpha;
        }
        0.
    }

    fn cdf(&self, x: f64) -> f64 {
        //! F(x) = P(X <= x)
        if x >= self.x_min {
            return 1. - (self.x_min / x).powf(self.alpha);
        }
        0.
    }

    fn ccdf(&self, x: f64) -> f64 {
        /*
        Complimentary cumulative distribution function (CCDF) also known as the survival function.
        */
        //! P(X > x) = 1 - F(x)
        1. - self.cdf(x)
    }

    fn rv(&self, u: f64) -> f64 {
        //! Random variate generation via inverse transform where u ~ Uniform(0,1).
        self.x_min / (1. - u).powf(1. / self.alpha)
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
