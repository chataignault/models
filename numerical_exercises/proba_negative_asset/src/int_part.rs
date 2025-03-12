pub fn all_partitions(n: u32) -> impl Iterator<Item = Vec<u32>> {
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
