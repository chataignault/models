use clap::Parser;
use std::error::Error;


#[derive(Parser, Debug)]
struct Args {
    n: u32,
    k: u32,
    a: f32,
    b: f32,
    p: f32
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    println!("n: {}\nk: {}\na,b: {}, {}\np: {}", args.n, args.k, args.a, args.b, args.p);
    Ok(())
}
