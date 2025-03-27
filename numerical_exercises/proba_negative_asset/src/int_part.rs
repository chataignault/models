pub fn all_partitions(n: u32) -> impl Iterator<Item = Vec<u32>> {
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

fn simple_ascending(n: u32) -> impl Iterator<Item = Vec<u32>> {
    // from https://jeromekelleher.net/generating-integer-partitions
    let mut part: Vec<Vec<u32>> = vec![];
    let mut a: Vec<u32> = vec![0; n.try_into().unwrap()];
    let mut k: usize = 1;
    a[1] = n;
    while k > 0 {
        let x = a[k - 1] + 1;
        let mut y = a[k] - 1;
        k -= 1;
        while x <= y {
            a[k] = x;
            y -= x;
            k += 1;
        }
        a[k] = x + y;
        part.push(a[..(k + 1)].to_vec());
    }
    part.into_iter()
}

fn efficient_ascending(n: u32) -> impl Iterator<Item = Vec<u32>> {
    // from https://jeromekelleher.net/generating-integer-partitions
    let mut part: Vec<Vec<u32>> = vec![];
    let mut a: Vec<u32> = vec![0; n.try_into().unwrap()];
    let mut k: usize = 1;
    let mut y: u32 = n - 1;
    while k > 0 {
        let mut x = a[k - 1] + 1;
        k -= 1;
        while 2 * x <= y {
            a[k] = x;
            y -= x;
            k += 1;
        }
        let l: usize = k + 1;
        while x <= y {
            a[k] = x;
            a[l] = y;
            part.push(a[..(k + 2)].to_vec());
            x += 1;
            y -= 1;
        }
        a[k] = x + y;
        y = x + y - 1;
        part.push(a[..(k + 1)].to_vec());
    }
    part.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASC_5: &[&[i32]] = &[
        &[1, 1, 1, 1, 1],
        &[1, 1, 1, 2],
        &[1, 1, 3],
        &[1, 2, 2],
        &[1, 4],
        &[2, 3],
        &[5],
    ];

    const ALL_3: &[&[i32]] = &[&[3], &[1, 2], &[2, 1], &[1, 1, 1]];

    #[test]
    fn test_integer_partition_naive() {
        // Expected partitions for n=3

        // Collect all generated partitions into a vector
        let generated_partitions: Vec<Vec<u32>> = all_partitions(3).collect();

        let expected_set: std::collections::HashSet<_> =
            ALL_3.iter().map(|p| format!("{:?}", p)).collect();

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

    #[test]
    fn test_integer_partition_simple() {
        // Expected partitions for n=5, in ascending order

        // Collect all generated partitions into a vector
        let generated_partitions: Vec<Vec<u32>> = simple_ascending(5).collect();

        let expected_set: std::collections::HashSet<_> =
            ASC_5.iter().map(|p| format!("{:?}", p)).collect();

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

    #[test]
    fn test_integer_partition_efficient() {
        // Expected partitions for n=5, in ascending order

        // Collect all generated partitions into a vector
        let generated_partitions: Vec<Vec<u32>> = efficient_ascending(5).collect();

        let expected_set: std::collections::HashSet<_> =
            ASC_5.iter().map(|p| format!("{:?}", p)).collect();

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
