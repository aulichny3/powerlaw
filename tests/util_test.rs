#[cfg(test)]
mod tests {
    use powerlaw::util;

    #[test]
    fn test_linspace() {
        let x = util::linspace(0.0, 1.0, 5);
        assert_eq!(x, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_sim_params() {
        let X: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let prec = 0.001;
        let xm = 78.;
        let params = util::sim::calculate_sim_params(&prec, X.as_slice(), &xm);

        assert_eq!(params.num_sims_m, 250000);
        assert_eq!(params.sim_len_n, X.len());
        assert_eq!(params.n_tail, (X.len() as f64 - xm) as usize);
    }

    #[test]
    fn test_check_data() {
        let mut X = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        X.push(-1.0);
        X.push(0.0);

        let z = util::check_data(&X);
        assert_eq!(z, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

}