//! An asynchronous pipeline, including a custom transport.
//!
//! ```sh
//! cargo run --example async_pipeline --features async -- testdata/imagenet-000000.tgz
//! ```
//!
//! The interesting part is [`SlowOpener`]: it stands in for an HTTP client by
//! serving shards from memory after a delay. With `concurrency(1)` the delays
//! add up; raising it overlaps them, which is the whole reason to be async.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::TryStreamExt;
use webdataset::asynch::{
    AsyncOpener, AsyncSampleStreamExt, AsyncShardSource, AsyncTupleStreamExt, AsyncWebDataset, BoxFuture,
};
use webdataset::{Error, Result};

/// Serves shards from memory after a delay, standing in for a network.
#[derive(Debug)]
struct SlowOpener {
    shards: HashMap<String, Arc<[u8]>>,
    latency: Duration,
}

impl AsyncOpener for SlowOpener {
    fn open<'a>(&'a self, url: &'a str) -> BoxFuture<'a, Result<AsyncShardSource>> {
        Box::pin(async move {
            // A real opener would await its HTTP client here; anything whose
            // body is a `Stream` of byte chunks becomes a shard source with
            // `futures_util::TryStreamExt::into_async_read`.
            sleep(self.latency).await;

            let bytes = self.shards.get(url).ok_or_else(|| Error::value(format!("no shard at {url}")))?;
            Ok(AsyncShardSource {
                url: url.to_string(),
                local_path: None,
                stream: Box::new(futures_util::io::Cursor::new(bytes.to_vec())),
            })
        })
    }
}

/// Sleep without pulling in a runtime, by parking a timer thread.
async fn sleep(duration: Duration) {
    let (sender, receiver) = futures_channel::oneshot::channel();
    std::thread::spawn(move || {
        std::thread::sleep(duration);
        let _ = sender.send(());
    });
    let _ = receiver.await;
}

fn main() -> Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: async_pipeline <shard>...");
        return Ok(());
    }

    // Load the shards once; the opener hands out copies as if fetching them.
    let mut shards = HashMap::new();
    let mut urls = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let url = format!("slow://shard-{i:06}.tar");
        shards.insert(url.clone(), Arc::<[u8]>::from(std::fs::read(path)?));
        urls.push(url);
    }
    let latency = Duration::from_millis(250);
    println!("{} shards, {latency:?} of latency each\n", urls.len());

    futures_executor::block_on(async {
        for concurrency in [1, urls.len().max(1)] {
            let opener = SlowOpener { shards: shards.clone(), latency };

            let dataset = AsyncWebDataset::builder_verbatim(urls.clone())
                .opener(Arc::new(opener))
                .concurrency(concurrency)
                .build()?
                .shuffle(1000)
                // Shards need not all hold the same fields, so keep the
                // samples this pipeline knows what to do with.
                .select(|sample| sample.get_first_spec("jpg;png").is_some())
                .decode_basic();

            let started = Instant::now();
            let (mut samples, mut batches) = (0usize, 0usize);

            let mut stream = dataset.stream().to_tuple(["jpg;png", "cls"]).batched(32, true);
            while let Some(batch) = stream.try_next().await? {
                samples += batch[1].as_tensor().map(|t| t.shape()[0]).unwrap_or(0);
                batches += 1;
            }

            println!("concurrency {concurrency}: {batches} batches, {samples} samples in {:.2?}", started.elapsed());
        }
        Ok(())
    })
}
