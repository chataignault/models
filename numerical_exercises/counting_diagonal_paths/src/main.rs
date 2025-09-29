use itertools::Itertools;

fn is_valid(p:Vec<u32>) -> bool {
    let mut valid = true;
    let mut prev = 0;
    let mut acc = 1;
    for i in &p {
        if *i == (prev + 1) && (prev != 0) {
            acc += 1;
            if acc == 3 {
                valid = false;
                break;
            }
        } else {
            acc = 1;
            if (*i - prev) > 3 {
                valid = false;
                break;
            }
        }
        prev = *i;
    }
    if 10 - prev >= 3 {
        valid = false;
    }
    if valid {
        println!("path : {:?}, {}", p, valid);
    }
    valid
}

fn main() {
    let range = 1..11;
    let n = 4;

    let mut total = 0;

    for path in range.combinations(n) {
        if is_valid(path) {
            total += 1;
        }
    }

    println!("Total number of valid paths is : {}", total);
}
