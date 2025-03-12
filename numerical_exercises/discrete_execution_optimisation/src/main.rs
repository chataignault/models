use clap::Parser;
use rand::rng;
use std::error::Error;
// use rand_distr::{Uniform, Distribution};
use rand_distr::uniform::{UniformFloat, UniformSampler};

#[derive(Parser, Debug)]
struct Args {
    n: u32,
    k: u32,
    a: f32,
    b: f32,
    p: f32,
}

fn expected_optimal_exec(n: u32, k: u32, a: f32, b: f32, p: f32) -> f32 {
    // Implement the optimal execution analytically
    if n == 0 {
        return 0.;
    } else if k == 0 || p >= b {
        return (b + a) as f32 / 2. * (n as f32);
    } else if k >= n && p <= a {
        return p * (n as f32);
    } else {
        let mut s: f32 = 0.;
        for i in 1..k {
            s += f32::min(p, a + ((n - i) as f32) * (b - a) / (n + 1) as f32);
        }
        for i in 1..(n - k) {
            s += a + (i as f32) * (b - a) / (n + 1) as f32
        }
        return s;
    }
}

fn exec_strategy_simulated(n: u32, k: u32, a: f32, b: f32, p: f32) -> f32 {
    // Numerically verify that the strategy is optimal
    const N: u32 = 10_000;
    let mut s: f32 = 0.;
    let t: f32 = a + ((n - k) as f32) * (b - a) / (n + 1) as f32;
    let mut rng_key = rng();
    let d = UniformFloat::<f32>::new(a, b).unwrap();
    for _ in 1..N {
        let mut sim: f32 = 0.;
        let mut r = k;
        for j in 1..n {
            let u = d.sample(&mut rng);
            if u > t && r > 0 {
                r -= 1;
                sim += p;
            } else {
                s += u;
            }
        }
        s += sim;
    }

    s / (N as f32)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    println!(
        "n: {}\nk: {}\na, b: {}, {}\np: {}",
        args.n, args.k, args.a, args.b, args.p
    );
    println!(
        "Expected : {}",
        expected_optimal_exec(args.n, args.k, args.a, args.b, args.p)
    );
    println!(
        "Strategy : {}",
        exec_strategy_simulated(args.n, args.k, args.a, args.b, args.p)
    );
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::*;

    const TOL: f32 = 1e-2;

    #[test]
    fn test_trivial_cases_optimal() {
        assert!((expected_optimal_exec(0, 1, 1., 2., 0.5).abs() < TOL));
        assert!(((expected_optimal_exec(3, 0, 1., 2., 0.5) - 4.5).abs() < TOL));
        assert!(((expected_optimal_exec(3, 2, 1., 2., 2.5) - 4.5).abs() < TOL));
        assert!(((expected_optimal_exec(3, 3, 1., 2., 0.5) - 1.5).abs() < TOL));
    }

    #[test]
    fn test_simulated_expected() {}

    #[test]
    fn test_sim_is_optimal() {
        assert!(
            (expected_optimal_exec(3, 2, 1., 2., 1.5) - exec_strategy_simulated(3, 2, 1., 2., 1.5))
                .abs()
                < TOL
        );
    }
}
