//! The snippets from the workspace README, kept compiling by the build.
//!
//! Running this would need a network, so `main` only checks that the pieces
//! type-check; the point of the example is that `cargo build --examples` fails
//! if the README drifts out of date.

#![allow(dead_code, unused_variables)]

use std::sync::Arc;

use webdataset::filters::{SampleIteratorExt, TupleIteratorExt};
use webdataset::{Decoder, DefaultEncoder, Result, Sample, ShardWriter, Value, WebDataset};

/// The "Quick start" reading snippet.
fn read() -> Result<()> {
    let dataset = WebDataset::builder("https://host/imagenet-{000000..000146}.tar")
        .shard_shuffle(100)
        .cache_dir("./_cache")
        .build()?
        .shuffle(1000)
        .decode(Decoder::default());

    for batch in dataset.iter().to_tuple(["jpg;png", "cls"]).batched(64, true) {
        let batch = batch?;
        let (images, labels) = (&batch[0], &batch[1]);
        // train on the batch
    }
    Ok(())
}

/// The "Quick start" writing snippet.
fn write() -> Result<()> {
    let mut writer =
        ShardWriter::new("train-%06d.tar")?.with_encoder(Arc::new(DefaultEncoder::new())).with_max_count(10_000);

    let mut sample = Sample::with_key("sample000001");
    sample.insert("cls", Value::Int(7));
    sample.insert("txt", Value::Text("a caption".into()));
    writer.write(&sample)?;
    writer.close()
}

/// The "Errors" snippet.
fn with_handler() -> Result<()> {
    let dataset =
        WebDataset::builder("shards-{000..999}.tar").handler(webdataset::handlers::warn_and_continue()).build()?;
    Ok(())
}

fn main() {
    println!("the README snippets compile");
}
