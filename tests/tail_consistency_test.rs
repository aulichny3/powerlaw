use powerlaw::dist::{exponential, pareto, Distribution};
use powerlaw::util::sim;

#[test]
fn test_tail_size_consistency_between_gof_and_sim() {
    //let mut X: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let mut X = vec![
        0.2574, 0.0197, 0.9573, 0.0868, 0.8581, 0.135, 0.4982, 2.0102, 1.2629, 2.2725,
    ];
    let n = X.len();
    let prec = 0.001;
    let xm = 0.8581;
    let a = 2.1542201760541153;

    // Step 1: test finding the MLE alphas for a given range of x_min and the data (Sec 3.1 Clauset et al.)
    let alphas = pareto::find_alphas_fast(&mut X);

    // Step 2: gof KS test the cdf of the proposed x_min and alpha hat vs the sample with x >= x_min. (Sec 3.3 Clauset et al.)
    // This is to find the best x_min/alpha pair given the data.
    let best_fit = dbg!(pareto::gof(&X, &alphas.0, &alphas.1));

    // Set up the hypothesis test simulation parameters.
    let params = dbg!(sim::calculate_sim_params(&prec, X.as_slice(), &xm));

    // Parameter the size of the tail should be the same as identified during parameter estimation and simulation setup.
    // It has been observed to not the case under some conditions...
    assert_eq!(best_fit.len_tail, params.n_tail);
}
