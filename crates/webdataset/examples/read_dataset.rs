//! Read a dataset and print a summary of what is in it.
//!
//! ```sh
//! cargo run --example read_dataset -- testdata/imagenet-000000.tgz
//! ```

use std::collections::BTreeMap;

use webdataset::{Result, WebDataset};

fn main() -> Result<()> {
    let shards: Vec<String> = std::env::args().skip(1).collect();
    if shards.is_empty() {
        eprintln!("usage: read_dataset <shard-pattern>...");
        eprintln!("  e.g. read_dataset 'testdata/imagenet-{{000000..000009}}.tgz'");
        return Ok(());
    }

    let dataset = WebDataset::builder_from(&shards).build()?.decode_basic();

    let mut count = 0usize;
    let mut fields: BTreeMap<String, usize> = BTreeMap::new();

    for sample in dataset.iter() {
        let sample = sample?;
        count += 1;
        for name in sample.field_names() {
            *fields.entry(name.to_string()).or_default() += 1;
        }
        if count <= 3 {
            println!("{}:", sample.key().unwrap_or("<no key>"));
            for (name, value) in sample.iter().filter(|(n, _)| !n.starts_with("__")) {
                println!("    {name:<12} {value:?}");
            }
        }
    }

    println!("\n{count} samples");
    for (name, seen) in &fields {
        println!("    {name:<12} in {seen} samples");
    }
    Ok(())
}
