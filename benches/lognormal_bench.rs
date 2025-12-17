use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use powerlaw::dist::lognormal::estimation::{
    lognormal_mle_truncated_par, lognormal_mle_truncated_serial,
};
use powerlaw::dist::lognormal::Lognormal;
use powerlaw::dist::Distribution;
use rand::prelude::*;

fn generate_lognormal_data(n: usize, mu: f64, sigma: f64) -> Vec<f64> {
    let dist = Lognormal::new(mu, sigma);
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            let u: f64 = rng.random();
            dist.rv(u)
        })
        .collect()
}

fn bench_lognormal_mle(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lognormal MLE");
    let mu = 0.5;
    let sigma = 0.2;
    let x_min = 1.0;

    // Test across different dataset sizes
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let data = generate_lognormal_data(*size, mu, sigma);

        group.bench_with_input(BenchmarkId::new("Serial", size), size, |b, &_| {
            b.iter(|| lognormal_mle_truncated_serial(black_box(&data), black_box(x_min)))
        });

        group.bench_with_input(BenchmarkId::new("Parallel", size), size, |b, &_| {
            b.iter(|| lognormal_mle_truncated_par(black_box(&data), black_box(x_min)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lognormal_mle);
criterion_main!(benches);
