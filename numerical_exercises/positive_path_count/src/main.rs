#![feature(test)]
extern crate test;

#[macro_use]
extern crate queues;

use clap::Parser;
use queues::{queue, IsQueue, Queue};
use std::collections::HashMap;
use std::error::Error;

use test::Bencher;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    length: u32,
}

fn brute_force(n: u32) -> u32 {
    // use a stack to keep track of the current admissible paths
    // upper bound for possible paths
    // FIFO ensures breadth-first search
    let mut N: u32 = 0;
    let mut q: Queue<(u32, u32)> = queue![(0, 0)];
    while q.size() > 0 {
        let (c, v) = q.remove().unwrap();
        if c == 2 * n && v == 0 {
            N += 1;
        } else if v == 0 {
            q.add((c + 1, v + 1));
        } else if v + c == 2 * n && v > 0 {
            q.add((c + 1, v - 1));
        } else {
            q.add((c + 1, v - 1));
            q.add((c + 1, v + 1));
        }
    }
    N
}

fn dynamical_programming(n: u32) -> u32 {
    // use strong recurrence relation
    let mut count_map: HashMap<(u32, u32), u32> = HashMap::new();
    count_map.insert((0, 0), 1);
    count_map.insert((0, 1), 0);
    for k in 1..(n + 1) {
        count_map.insert((k, k), 1);
        count_map.insert(
            (k, 0),
            *count_map.get(&(k - 1, 1)).unwrap() + *count_map.get(&(k - 1, 0)).unwrap(),
        );
        if k > 1 {
            for j in 1..(k - 1) {
                count_map.insert(
                    (k, j),
                    2 * (*count_map.get(&(k - 1, j)).unwrap())
                        + *count_map.get(&(k - 1, j - 1)).unwrap()
                        + *count_map.get(&(k - 1, j + 1)).unwrap(),
                );
            }
            count_map.insert(
                (k, k - 1),
                2 * (*count_map.get(&(k - 1, k - 1)).unwrap())
                    + *count_map.get(&(k - 1, k - 2)).unwrap(), // + *count_map.get(&(k - 1, j + 1)).unwrap(),
            );
        }
    }
    *count_map.get(&(n, 0)).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    println!("Input : {}", args.length);

    println!("Brute force result : {}", brute_force(args.length));
    println!(
        "Dynamic programming result : {}",
        dynamical_programming(args.length)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf_init() {
        assert_eq!(brute_force(0), 1);
        assert_eq!(brute_force(1), 1);
        assert_eq!(brute_force(2), 2);
        assert_eq!(brute_force(3), 5);
    }

    #[test]
    fn test_dyn_init() {
        assert_eq!(dynamical_programming(0), 1);
        assert_eq!(dynamical_programming(1), 1);
        assert_eq!(dynamical_programming(2), 2);
        assert_eq!(dynamical_programming(3), 5);
    }

    #[test]
    fn test_match_results() {
        for n in 6..10 {
            assert_eq!(brute_force(n), dynamical_programming(n));
        }
    }

    #[bench]
    fn bench_exec_time_dyn(b: &mut Bencher) {
        b.iter(|| dynamical_programming(5));
    }

    #[bench]
    fn bench_exec_time_bf(b: &mut Bencher) {
        b.iter(|| brute_force(5));
    }
}
