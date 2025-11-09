use criterion::{criterion_group, criterion_main, Criterion};
use powerlaw::{dist, stats, util, Distribution};
use std::hint::black_box;

fn alpha_mle_benchmark(c: &mut Criterion) {
    // read data from csv into vector and sum it
    let mut data = util::read_csv("Data/reference_data/blackouts.txt").unwrap();

    c.bench_function("fast alphas abs neg rets", |b| {
        b.iter(|| dist::pareto::find_alphas_fast(&mut data))
    });
}

fn alpha_mle_benchmark_all(c: &mut Criterion) {
    let mut data = util::read_csv("Data/reference_data/blackouts.txt").unwrap();
    let mut group = c.benchmark_group("Alpha MLE");

    group.bench_function("fast", |b| {
        b.iter(|| dist::pareto::find_alphas_fast(&mut data))
    });

    group.bench_function("exhaustive", |b| {
        b.iter(|| dist::pareto::find_alphas_exhaustive(&mut data))
    });

    group.finish();
}

fn ks_test_benchmark(c: &mut Criterion) {
    let xm: f64 = 43.0;
    let a: f64 = 0.88;

    // generate 500 U(0,1)
    let U = stats::random::random_uniform(500);
    // generate pareto type I distributed data
    let mut X: Vec<f64> = U
        .iter()
        .map(|u| {
            dist::pareto::Pareto {
                x_min: xm,
                alpha: a,
            }
            .rv(*u)
        })
        .collect();

    // sort
    X.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let custom_cdf = |x: f64| {
        dist::pareto::Pareto {
            x_min: xm,
            alpha: a,
        }
        .cdf(x)
    };
    c.bench_function("KS test of p=Pareto Type I x_min=43.0 alpha = 0.88", |b| {
        b.iter(|| stats::ks::ks_1sam_sorted(&X, custom_cdf))
    });
}

criterion_group!(benches, ks_test_benchmark, alpha_mle_benchmark_all);
criterion_main!(benches);
