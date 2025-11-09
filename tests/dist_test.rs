use powerlaw::dist::{exponential, pareto, Distribution};

#[test]
fn random_variate() {
    let expo = exponential::Exponential { lambda: 4.0 };

    // X = x
    let X: f64 = expo.rv(0.2);

    assert_eq!(X, 0.40235947810852507);
}

#[test]
#[should_panic]
fn pareto_params() {
    let invalid = pareto::Pareto::new(-1., -1.);
}
