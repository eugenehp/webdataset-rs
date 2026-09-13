//! Build a small dataset, then read it back.
//!
//! ```sh
//! cargo run --example write_dataset -- /tmp/demo
//! ```

use std::sync::Arc;

use webdataset::filters::SampleIteratorExt;
use webdataset::{DefaultEncoder, Result, Sample, ShardWriter, Tensor, Value, WebDataset};

fn main() -> Result<()> {
    let directory = std::env::args().nth(1).unwrap_or_else(|| "./demo-dataset".to_string());
    std::fs::create_dir_all(&directory)?;

    // Each shard holds at most 250 samples, so 1000 samples make four shards.
    let pattern = format!("{directory}/demo-%06d.tar");
    let mut writer = ShardWriter::new(&pattern)?.with_encoder(Arc::new(DefaultEncoder::new())).with_max_count(250);

    for i in 0..1000i64 {
        let mut sample = Sample::with_key(format!("item{i:06}"));
        sample.insert("cls", Value::Int(i % 10));
        sample.insert("txt", Value::Text(format!("item number {i}")));
        sample.insert("npy", Value::Tensor(Tensor::from_f32(&[i as f32, (i % 10) as f32])));
        writer.write(&sample)?;
    }
    let shards = writer.shard_count();
    writer.close()?;
    println!("wrote 1000 samples into {shards} shards under {directory}");

    // Read them back, shuffled and batched.
    let dataset = WebDataset::builder(format!("{directory}/demo-{{000000..{:06}}}.tar", shards - 1))
        .shard_shuffle(4)
        .seed(0)
        .build()?
        .shuffle(200)
        .decode_basic();

    let mut batches = 0usize;
    for batch in dataset.iter().batched(64, true) {
        let batch = batch?;
        if batches == 0 {
            let classes = batch.get("cls").and_then(Value::as_tensor).expect("cls column");
            let vectors = batch.get("npy").and_then(Value::as_tensor).expect("npy column");
            println!("first batch: cls {:?}, npy {:?}", classes.shape(), vectors.shape());
        }
        batches += 1;
    }
    println!("read back {batches} batches");
    Ok(())
}
