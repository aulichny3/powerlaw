use powerlaw::dist::lognormal::estimation::lognormal_mle_truncated_serial;
use powerlaw::dist::lognormal::Lognormal;
use powerlaw::dist::Distribution;
use rand::Rng;
#[test]
fn test_lognormal_mle_truncation_validity() {
    // 1. Setup True Parameters
    // choose parameters that create a distinct shape.
    let true_mu = 0.5;
    let true_sigma = 0.8;
    // choose a truncation point x_min.
    // Lognormal(0.5, 0.8) has mode at exp(0.5 - 0.8^2) = exp(0.5 - 0.64) = exp(-0.14) ~= 0.87
    // Median is exp(0.5) ~= 1.64
    // x_min = 1.0 is a reasonable truncation point (somewhere in the body).
    let x_min = 1.0;

    let n_samples = 100000;

    let true_dist = Lognormal::new(true_mu, true_sigma);

    // 2. Generate Data
    // generate data from the true distribution and keep only values >= x_min
    let mut data = Vec::with_capacity(n_samples);
    let mut rng = rand::rng();

    // Safety break to prevent infinite loop if params are bad
    let max_tries = n_samples * 10;
    let mut tries = 0;

    while data.len() < n_samples && tries < max_tries {
        let u: f64 = rng.random();
        let x = true_dist.rv(u);
        if x >= x_min {
            data.push(x);
        }
        tries += 1;
    }

    if data.len() < n_samples {
        panic!("Could not generate enough samples exceeding x_min");
    }

    // 3. Estimate Parameters
    // use the serial version as the logic is identical to parallel for the math.
    let (est_mu, est_sigma) = lognormal_mle_truncated_serial(&data, x_min);

    println!("True mu: {}, Est mu: {}", true_mu, est_mu);
    println!("True sigma: {}, Est sigma: {}", true_sigma, est_sigma);

    // 4. Verify Parameters
    // expect the MLE to recover the *underlying* parameters mu and sigma.
    // With 100k samples, the error should be small (< 0.02 is a conservative bound).
    assert!(
        (est_mu - true_mu).abs() < 0.02,
        "Estimated mu ({}) deviated too much from true mu ({})",
        est_mu,
        true_mu
    );
    assert!(
        (est_sigma - true_sigma).abs() < 0.02,
        "Estimated sigma ({}) deviated too much from true sigma ({})",
        est_sigma,
        true_sigma
    );

    // 5. Verify Density Validity (Integration)
    // We use the new_truncated constructor which handles normalization internally.
    let est_dist = Lognormal::new_truncated(est_mu, est_sigma, x_min);

    // Integrate the *truncated* PDF from x_min to infinity.
    // Since the struct now handles truncation, est_dist.pdf(x) is already f_trunc(x).

    let mut area = 0.0;
    // Step size for integration
    let step = 0.005;
    // Upper limit for integration (effectively infinity).
    // exp(mu + 8*sigma) covers almost all probability mass.
    let limit = (est_mu + 8.0 * est_sigma).exp();

    let mut x = x_min;
    while x < limit {
        let p_mid = est_dist.pdf(x + step / 2.0);
        area += p_mid * step;
        x += step;
    }

    println!(
        "Integral of truncated PDF (using struct internal normalization): {}",
        area
    );

    // The integral should be very close to 1.0.
    assert!(
        (area - 1.0).abs() < 0.01,
        "The truncated PDF (from Lognormal struct) does not integrate to 1.0. Integral: {}",
        area
    );
}
