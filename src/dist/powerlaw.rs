// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! The generic powerlaw distribution defined by Cf(x) where f(x) = x^(-alpha) and C is the normalizing constant to ensure the distribution integrates to 1
//! where C = (alpha - 1) * x_min^(alpha - 1).
//! Continuous, Unbounded Power Law which simplifies to a pareto type I.
//! However, the alpha parameter will be exactly 1.0 greater than that of more
//! common expressions such as the pareto type I pdf listed at:
//! [https://en.wikipedia.org/wiki/Pareto_distribution](https://en.wikipedia.org/wiki/Pareto_distribution)
use super::Distribution;
use rand::prelude::*;

/// Powerlaw struct with parameters alpha and x_min.
pub struct Powerlaw {
    pub alpha: f64,
    pub x_min: f64,
}

impl Distribution for Powerlaw {
    /// Continuous, Unbounded Power Law which simplifies to a pareto type I.
    /// The alpha parameter will be exactly 1.0 greater than that of more
    /// common expressions such as the pdf listed at:
    /// <https://en.wikipedia.org/wiki/Pareto_distribution>
    fn pdf(&self, x: f64) -> f64 {
        (self.alpha - 1.) / self.x_min.powf(1. - self.alpha) * x.powf(-self.alpha)
    }

    fn cdf(&self, x: f64) -> f64 {
        /*
        Continuous, Unbounded Power Law
         */
        1. - (x / self.x_min).powf(-self.alpha + 1.)
    }

    fn ccdf(&self, x: f64) -> f64 {
        1. - self.cdf(x)
    }

    fn rv(&self, u: f64) -> f64 {
        /*
        Random variate generation via inverse transform of the pareto type 1 pdf

        Parameters
        ----------
        u: f64
            Random Uniform (0,1)

         */

        self.x_min * (1. - u).powf(-1. / (self.alpha - 1.))
    }
}

/// MLE estimator for the alpha parameter.
pub fn alpha_hat(data: &[f64], x_min: f64) -> f64 {
    let n: usize = data.len();

    let logs = data.iter().map(|x: &f64| (x / x_min).ln());
    let sum_of_logs: f64 = logs.sum();

    1. + n as f64 / sum_of_logs
}
