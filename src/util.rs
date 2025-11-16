// Copyright (c) 2025 Adam Ulichny
//
// This source code is licensed under the MIT OR Apache-2.0 license
// that can be found in the LICENSE-MIT or LICENSE-APACHE files
// at the root of this source tree.

//! Module and its submodules containing various helper functions primarily around generating synthetic datasets.

use csv::ReaderBuilder;
use std::error::Error;

/// Returns *n* quantity of evenly spaced numbers over a specified interval. Motivated by numpy's [linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html).
///
/// # Parameters
/// - `start`: The starting value of the sequence.
/// - `end`: The ending value of the sequence.
/// - `n`: The number of samples to generate.
///
/// # Example
/// ```
/// use powerlaw::util;
///
/// let numbers:Vec<f64> = util::linspace(0.,1., 5); // results in [0.0, 0.25, 0.5, 0.75, 1.0]
/// ```
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| start + i as f64 * (end - start) / (n - 1) as f64)
        .collect()
}

/// Functions for handling simulation related tasks such as identifying recommended parameters and generating synthetic datasets.
pub mod sim {
    use crate::dist::pareto::Pareto;
    use crate::dist::Distribution;
    use crate::stats::random::random_uniform;
    use rand::prelude::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// Contains the number of simulations, quantity of elements in each sim, size of the distribution tail, and probability of the tail event.
    #[derive(Debug)]
    pub struct SimParams {
        pub num_sims_m: usize,
        pub sim_len_n: usize,
        pub n_tail: usize,
        pub p_tail: f64,
    }

    /// Calculates the number of simulations, number of samples per sim, the size of the tail given a predetermined x_min, and calculate the probability of the tail event.
    /// The methodology is based on what is proposed in Section 4.1 of Clauset, Aaron, et al. ‘Power-Law Distributions in Empirical Data’.
    /// SIAM Review, vol. 51, no. 4, Society for Industrial & Applied Mathematics (SIAM), Nov. 2009, pp. 661–703, [doi:10.48550/ARXIV.0706.1062](https://doi.org/10.48550/arXiv.0706.1062).
    /// Where the number of simulations required for the desired level of precision in the estimate is: 1/4 * prec^(-2). Ex. 1/4 * 0.01^(-2) = 2500 sims gives accuracy within 0.01
    ///
    /// # Example
    /// ```
    /// use powerlaw::util;
    ///
    /// let X: Vec<f64> = (0..100).map(|x| x as f64).collect();
    /// let prec = 0.01;
    /// let xm = 78.;
    /// let params = util::sim::calculate_sim_params(&prec, X.as_slice(), &xm); // params.num_sims_m = 2500
    /// ```
    pub fn calculate_sim_params(prec: &f64, data: &[f64], x_min: &f64) -> SimParams {
        // calculate number of sims based on desired precision
        let M: usize = ((1. / 4.) * prec.powf(-2.)).round() as usize;

        // sample size per sim exhaustive
        let n: &usize = &data.len();

        // determine the size of the tail
        // Use a chain of iterator methods
        let n_tail: &usize = &data.iter().filter(|&&x| x >= *x_min).count();
        //probability of the tail
        let p_tail: f64 = *n_tail as f64 / *n as f64;

        SimParams {
            num_sims_m: M,
            sim_len_n: *n,
            n_tail: *n_tail,
            p_tail: p_tail,
        }
    }

    /// Generates multiple synthetic datasets using a hybrid model based on the input data and a proposed Pareto Type I fit. This process is fully parallelized,
    /// with M simulations running concurrently on separate threads.
    ///
    /// Each simulated dataset (of size 'n') is constructed by mixing two sampling mechanisms:
    /// 1. Sampling from the 'lower' part of the original data (where x < x_min).
    /// 2. Sampling from a Pareto Type I distribution (defined by x_min and alpha).
    ///
    /// The probability of selecting from the Pareto tail is controlled by `p_tail`.
    ///
    /// This approach is commonly used in bootstrapping or simulation studies for extreme value analysis.
    pub fn generate_synthetic_datasets(
        data: &[f64],
        x_min: f64,
        sim_params: SimParams,
        alpha: f64,
    ) -> Vec<Vec<f64>> {
        // create a vector of all the data < the tail
        let lower: Vec<f64> = data.iter().filter(|&&x| x < x_min).copied().collect();

        // make the simulations
        // Create an arc of the lower sample data and take ownership of it
        let shared_lower: Arc<Vec<f64>> = Arc::new(lower);

        // create one arc<Mutex> to share the vector of vectors
        let sims: Arc<Mutex<Vec<Vec<f64>>>> = Arc::new(Mutex::new(Vec::new()));

        // create the handles vector to store the results of the spawned threads
        let mut handles = vec![];

        // generate M simulated datasets of size n
        for _i in 0..sim_params.num_sims_m {
            //create the threads
            let handle = thread::spawn({
                // clone the arc for each thread to increase the reference count
                let shared_lower_vec = shared_lower.clone();

                let sims_clone = sims.clone();
                move || {
                    // create empty vector of known size to store synth data set
                    let mut synth: Vec<f64> = Vec::new();
                    synth.reserve_exact(sim_params.sim_len_n);

                    // Get an RNG:
                    let mut rng = rand::rng();

                    // make a vector of random U(0,1) equal to the length of the sample data.
                    //let r = random_uniform(*shared_n_const);
                    let r = random_uniform(sim_params.sim_len_n);

                    for u in r {
                        if u < sim_params.p_tail {
                            //let pareto_dist = Pareto {x_min: x_min, alpha: alpha};
                            //let x = pareto_dist.rv(&mut rng, 1)[0];

                            let x = Pareto {
                                x_min: x_min,
                                alpha: alpha,
                            }
                            .rv(rng.random());

                            synth.push(x);
                            continue;
                        }
                        let x = &shared_lower_vec.choose(&mut rng).unwrap();
                        //println!("random sample is {x}");
                        synth.push(**x);
                    }

                    let mut sims_results = sims_clone.lock().unwrap();
                    sims_results.push(synth);
                }
            });

            handles.push(handle);
        }
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let final_sim = sims.lock().unwrap();

        final_sim.clone()
    }
}

/// Reads a single-column CSV file into a vector of `f64`.
///
/// This function assumes the CSV has no headers. It will skip any rows
/// that cannot be parsed into an `f64`.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
pub fn read_csv(file_path: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_path)?;

    let data: Vec<f64> = rdr
        .records()
        .filter_map(|result| result.ok())
        .filter_map(|record| record.get(0)?.parse::<f64>().ok())
        .collect();

    Ok(data)
}

/// Filters a slice to ensure all values are positive.
///
/// This function checks for non-positive values (`<= 0.0`) or `NaN`. If such
/// values are found, it prints a warning to `stderr` and returns a new `Vec<f64>`
/// containing only the positive values. Otherwise, it returns a clone of the
/// original slice.
pub fn check_data(data: &[f64]) -> Vec<f64> {
    if data.iter().any(|&x| !(x > 0.0)) {
        eprintln!("Warning: Data contains non-positive values or NaN. Filtering applied.");

        // Filter and return the new Vec
        data.iter().filter(|&&x| x > 0.0).copied().collect()
    } else {
        // Clone and return a copy of the original Vec
        data.to_vec()
    }
}
