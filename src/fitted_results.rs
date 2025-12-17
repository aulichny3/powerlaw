// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Keeps track of fitted distributions, their names and parameters.
use crate::dist::Distribution;

/// A container to hold and manage multiple fitted distributions.
pub struct FittedResults {
    distributions: Vec<Box<dyn Distribution>>,
}

impl FittedResults {
    /// Creates a new, empty collection.
    pub fn new() -> Self {
        Self {
            distributions: Vec::new(),
        }
    }

    /// Adds a new fitted distribution to the collection.
    /// The distribution must implement the `Distribution` trait and be valid for the
    /// lifetime of the program ('static).
    pub fn add<D: Distribution + 'static>(&mut self, dist: D) {
        self.distributions.push(Box::new(dist));
    }

    /// Returns a summary of all fitted distributions and their parameters.
    pub fn summary(&self) -> Vec<(&'static str, Vec<(&'static str, f64)>)> {
        self.distributions
            .iter()
            .map(|d| (d.name(), d.parameters()))
            .collect()
    }
}
