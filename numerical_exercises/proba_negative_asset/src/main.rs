#[macro_use]
extern crate approx;

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

fn compute_proba_negative(n: u32, p: f64, s: f32, ds: f32) -> f64 {
    let r: i32 = (s / ds).ceil() as i32;
    println!("r={}", r);
    if r <= 0 {
        return 1.;
    } else if n as i32 - r < 0 {
        return 0.;
    } else {
        let mut p_hat: f64 = (1. - p).powf(r.into());
        println!("p0={}", p_hat);
        let max_steps_up: i32 = (n as i32 - r) / 2 + 1;
        println!("max up {}", max_steps_up);
        if max_steps_up >= 1 {
            p_hat += p
                * (1. - p).powf((r + 1 as i32).into())
                * ((binomial(2, 1) - binomial(2, 2)) * r) as f64;
        }
        for k in 2..max_steps_up {
            p_hat += p.powf(k.into())
                * (1. - p).powf((r + k as i32).into())
                * ((binomial(2 * k, k) - binomial(2 * k, k + 1)) * r
                    + (binomial(2 * (k - 1), k - 1) - binomial(2 * (k - 1), k)) * (r - 1)
                    + 1) as f64;
            println!("ph={}", p_hat);
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

    const TOL: f64 = 1e-3;

    #[test]
    fn test_mc_trivial_cases() {
        _ = abs_diff_eq!(
            compute_proba_negative_mc(1, 0.5, -39., 20.),
            1.,
            epsilon = TOL
        );
        _ = abs_diff_eq!(
            compute_proba_negative_mc(1, 0.5, 100., 20.),
            0.,
            epsilon = TOL
        );
        _ = abs_diff_eq!(
            compute_proba_negative_mc(1, 0.5, 50., 60.),
            0.5,
            epsilon = TOL
        );
        _ = abs_diff_eq!(
            compute_proba_negative_mc(0, 0.5, 50., 60.),
            0.,
            epsilon = TOL
        );
        _ = abs_diff_eq!(
            compute_proba_negative_mc(0, 0.5, -50., 60.),
            1.,
            epsilon = TOL
        );
    }

    #[test]
    fn test_analytic_trivial_cases() {
        _ = abs_diff_eq!(compute_proba_negative(1, 0.5, -39., 20.), 1., epsilon = TOL);
        _ = abs_diff_eq!(compute_proba_negative(1, 0.5, 100., 20.), 0., epsilon = TOL);
        _ = abs_diff_eq!(compute_proba_negative(1, 0.5, 50., 60.), 0.5, epsilon = TOL);
        _ = abs_diff_eq!(compute_proba_negative(0, 0.5, 50., 60.), 0., epsilon = TOL);
        _ = abs_diff_eq!(compute_proba_negative(0, 0.5, -50., 60.), 1., epsilon = TOL);
    }

    #[test]
    fn test_approx_equal() {
        _ = abs_diff_eq!(
            compute_proba_negative(10, 0.5, 100., 20.),
            compute_proba_negative_mc(10, 0.5, 100., 20.),
            epsilon = TOL
        );
        _ = abs_diff_eq!(
            compute_proba_negative(20, 0.4, 350., 39.),
            compute_proba_negative_mc(20, 0.4, 350., 39.),
            epsilon = TOL
        );
        _ = abs_diff_eq!(
            compute_proba_negative(30, 0.3, 350., 39.),
            compute_proba_negative_mc(30, 0.3, 350., 39.),
            epsilon = TOL
        );
    }
}
