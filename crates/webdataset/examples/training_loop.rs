//! The shape of a training loop: shuffled, decoded, batched, multi-worker.
//!
//! ```sh
//! cargo run --example training_loop -- testdata/imagenet-000000.tgz
//! ```

use std::time::Instant;

use webdataset::filters::{SampleIteratorExt, TupleIteratorExt};
use webdataset::{Result, WebDataset, handlers};

const BATCH_SIZE: usize = 32;
const EPOCH_SAMPLES: usize = 512;
const WORKERS: usize = 4;

fn main() -> Result<()> {
    let shards: Vec<String> = std::env::args().skip(1).collect();
    if shards.is_empty() {
        eprintln!("usage: training_loop <shard-pattern>...");
        return Ok(());
    }

    let dataset = WebDataset::builder_from(&shards)
        // Resampling means every worker always has shards to read, whatever
        // the ratio of shards to workers.
        .resampled(true)
        .shard_shuffle(100)
        .seed(0)
        // Streaming a large dataset will meet some unreadable bytes; log them
        // and keep training rather than dying half way through an epoch.
        .handler(handlers::warn_and_continue())
        .build()?
        .shuffle(1000)
        .decode_basic()
        // A fixed epoch length keeps the training loop simple. Each worker
        // runs its own copy of the pipeline, so the limit is per worker and
        // has to be divided to get the epoch size you asked for.
        .with_epoch(EPOCH_SAMPLES / WORKERS);

    let loader = dataset.loader().with_workers(WORKERS).with_prefetch(16);

    for epoch in 0..3 {
        let started = Instant::now();
        let mut samples = 0usize;
        let mut batches = 0usize;

        for batch in loader.iter().to_tuple(["jpg;png", "cls"]).batched(BATCH_SIZE, false) {
            let batch = batch?;
            let (_images, labels) = (&batch[0], &batch[1]);
            samples += labels.as_tensor().map(|t| t.shape()[0]).unwrap_or(BATCH_SIZE);
            batches += 1;
        }

        let elapsed = started.elapsed();
        println!(
            "epoch {epoch}: {batches} batches, {samples} samples in {:.2?} ({:.0} samples/s)",
            elapsed,
            samples as f64 / elapsed.as_secs_f64()
        );
    }
    Ok(())
}
