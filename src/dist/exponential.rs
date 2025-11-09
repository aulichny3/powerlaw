// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Exponential distribution parameterized by its lambda parameter.

use super::Distribution;

pub struct Exponential {
    pub lambda: f64,
}

impl Distribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        //! f(x)dx = 1

        self.lambda * std::f64::consts::E.powf(-self.lambda * x)
    }

    fn cdf(&self, x: f64) -> f64 {
        //! F(x) = P(X <= x)
        1. - std::f64::consts::E.powf(-self.lambda * x)
    }

    fn ccdf(&self, x: f64) -> f64 {
        //! P(X > x) = 1 - F(x)
        1. - self.cdf(x)
    }

    fn rv(&self, u: f64) -> f64 {
        //! Random variate generation via inverse transform where u ~ Uniform(0,1).
        -u.ln() / self.lambda
    }
}
