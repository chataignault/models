use clap::Parser;
use std::error::Error;

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
        0.
    }
}

fn exec_strategy_simulated(n: u32, k: u32, a: f32, b: f32, p: f32) -> f32 {
    // Numerically verify that the strategy is optimal
    0.
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

    const TOL: f32 = 1e-3;

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
