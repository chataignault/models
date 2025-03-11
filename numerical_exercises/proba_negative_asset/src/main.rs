use clap::Parser;
use num_integer::binomial;
use rand::distr::{Bernoulli, Distribution};
use std::error::Error;

#[derive(Parser, Debug)]
struct Args {
    n: u32,
    p: f64,
    s: f32,
    x: f32,
}

fn compute_proba_negative_mc(n: u32, p: f64, s: f32, ds: f32) -> f64 {
    // Monte Carlo estimate of the path-dependant probability
    let r: i32 = (s / ds).ceil() as i32;
    let mut rng = rand::rng();
    let d = Bernoulli::new(p).unwrap();
    const N: u64 = 20_000;
    let mut p_hat: u64 = 0;
    for _ in 0..N {
        let mut sim: Vec<f64> = d
            .sample_iter(&mut rng)
            .take(usize::try_from(n).unwrap())
            .collect::<Vec<bool>>()
            .iter()
            .map(|&e| 2. * (e as i64 as f64) - 1.)
            .collect();
        p_hat += sim
            .iter_mut()
            .fold((r <= 0, r as f64), |acc, x| {
                *x += acc.1;
                (acc.0 || (*x <= 0.), *x)
            })
            .0 as u64;
    }
    p_hat as f64 / N as f64
}

fn all_partitions(n: u32) -> impl Iterator<Item = Vec<u32>> {
    // Generate all possible partitions first
    fn generate_partitions(n: u32) -> Vec<Vec<u32>> {
        if n == 0 {
            return vec![vec![]];
        }
        if n == 1 {
            return vec![vec![1]];
        }

        let mut result = Vec::new();

        // Try each possible first part
        for first in 1..=n {
            // Recursively generate all partitions of the remainder
            let sub_partitions = generate_partitions(n - first);

            for partition in sub_partitions {
                // Try inserting 'first' at each possible position
                for i in 0..=partition.len() {
                    let mut new_partition = partition.clone();
                    new_partition.insert(i, first);
                    result.push(new_partition);
                }
            }
        }

        // Deduplicate
        result.sort();
        result.dedup();
        result
    }

    let partitions = generate_partitions(n);
    let mut index = 0;

    std::iter::from_fn(move || {
        if index < partitions.len() {
            let result = partitions[index].clone();
            index += 1;
            Some(result)
        } else {
            None
        }
    })
}

fn catalan(k: u32) -> u32 {
    binomial(2 * k, k) - binomial(2 * k, k + 1)
}

fn compute_proba_negative(n: u32, p: f64, s: f32, ds: f32) -> f64 {
    let r: i32 = (s / ds).ceil() as i32;
    if r <= 0 {
        return 1.;
    } else if n as i32 - r < 0 {
        return 0.;
    } else {
        let mut p_hat: f64 = (1. - p).powf(r.into());
        let max_steps_up: i32 = (n as i32 - r) / 2 + 1;

        for k in 1..max_steps_up {
            let pk = p.powf(k.into()) * (1. - p).powf((r + k as i32).into());
            let partitions = all_partitions(k as u32);
            let mut path_count: u32 = 0;
            for partition in partitions {
                let s = partition.len();
                let pc: u32 = partition
                    .iter()
                    .map(|j| catalan(*j as u32))
                    .product::<u32>()
                    * binomial(r as u32, u32::try_from(s).unwrap_or(0));
                path_count += pc;
            }
            p_hat += pk * path_count as f64;
        }

        return p_hat;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Analytic solution using first principles reasoning
    // using the stopping time of going lower than initially by one step
    let args = Args::parse();
    println!(
        "Steps: {}\nProba up: {}, \nStart value: {}, \nPrice step: {}",
        args.n, args.p, args.s, args.x
    );
    println!(
        "Monte Carlo estimate: {}",
        compute_proba_negative_mc(args.n, args.p, args.s, args.x)
    );
    println!(
        "Analytic value: {}",
        compute_proba_negative(args.n, args.p, args.s, args.x)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-2;

    #[test]
    fn test_mc_trivial_cases() {
        assert!((compute_proba_negative_mc(1, 0.5, -39., 20.) - 1.).abs() < TOL);
        assert!(compute_proba_negative_mc(1, 0.5, 100., 20.).abs() < TOL);
        assert!((compute_proba_negative_mc(1, 0.5, 50., 60.) - 0.5).abs() < TOL);
        assert!(compute_proba_negative_mc(0, 0.5, 50., 60.).abs() < TOL);
        assert!((compute_proba_negative_mc(0, 0.5, -50., 60.) - 1.).abs() < TOL);
    }

    #[test]
    fn test_analytic_trivial_cases() {
        assert!((compute_proba_negative(1, 0.5, -39., 20.) - 1.).abs() < TOL);
        assert!(compute_proba_negative(1, 0.5, 100., 20.).abs() < TOL);
        assert!((compute_proba_negative(1, 0.5, 50., 60.) - 0.5).abs() < TOL);
        assert!(compute_proba_negative(0, 0.5, 50., 60.).abs() < TOL);
        assert!((compute_proba_negative(0, 0.5, -50., 60.) - 1.).abs() < TOL);
    }

    #[test]
    fn test_approx_equal() {
        assert!(
            compute_proba_negative(10, 0.5, 100., 20.)
                - compute_proba_negative_mc(10, 0.5, 100., 20.)
                < TOL
        );
        assert!(
            compute_proba_negative(20, 0.4, 350., 39.)
                - compute_proba_negative_mc(20, 0.4, 350., 39.)
                < TOL
        );
        assert!(
            compute_proba_negative(30, 0.3, 350., 39.)
                - compute_proba_negative_mc(30, 0.3, 350., 39.)
                < TOL
        );
    }

    #[test]
    fn test_integer_partition() {
        // Expected partitions for n=3
        let expected_partitions = vec![vec![3], vec![1, 2], vec![2, 1], vec![1, 1, 1]];

        // Collect all generated partitions into a vector
        let generated_partitions: Vec<Vec<u32>> = all_partitions(3).collect();

        // Test 1: Check that all expected partitions are generated
        for expected in &expected_partitions {
            assert!(
                generated_partitions.contains(expected),
                "Missing expected partition: {:?}",
                expected
            );
        }

        // Test 2: Check that no extra partitions are generated
        assert_eq!(
            generated_partitions.len(),
            expected_partitions.len(),
            "Generated partitions count doesn't match expected: {:?}",
            generated_partitions
        );

        // Alternative approach using sets for more readable error messages
        let expected_set: std::collections::HashSet<_> = expected_partitions
            .iter()
            .map(|p| format!("{:?}", p))
            .collect();

        let generated_set: std::collections::HashSet<_> = generated_partitions
            .iter()
            .map(|p| format!("{:?}", p))
            .collect();

        // Check for missing partitions
        let missing: Vec<_> = expected_set.difference(&generated_set).collect();
        assert!(missing.is_empty(), "Missing partitions: {:?}", missing);

        // Check for extra partitions
        let extra: Vec<_> = generated_set.difference(&expected_set).collect();
        assert!(extra.is_empty(), "Extra partitions: {:?}", extra);
    }
}
