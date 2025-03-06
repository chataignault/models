use clap::Parser;
use queues::{queue, IsQueue, Queue};
use std::collections::HashMap;
use std::error::Error;

#[macro_use]
extern crate queues;

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
    if n == 1 {
        return 1;
    }
    if n == 2 {
        return 2;
    }
    n
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
}
