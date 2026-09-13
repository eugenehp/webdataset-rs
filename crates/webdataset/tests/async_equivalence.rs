//! The asynchronous pipeline must read exactly what the blocking one reads.
//!
//! Everything above the transport is shared between the two, so any divergence
//! would be in the archive parser or the stream plumbing — which is precisely
//! what these tests exercise, shard by shard and decoder by decoder.

#![cfg(feature = "async")]

use std::sync::Arc;

use futures_executor::block_on;
use futures_util::{StreamExt, TryStreamExt};
use webdataset::asynch::{AsyncSampleStreamExt, AsyncTupleStreamExt, AsyncWebDataset, MemoryOpener};
use webdataset::filters::{SampleIteratorExt, TupleIteratorExt};
use webdataset::{Decoder, Sample, Value, WebDataset};

/// Every bundled shard, so no format goes unchecked.
const SHARDS: &[&str] =
    &["sample.tgz", "imagenet-000000.tgz", "mpdata.tar", "tendata.tar", "testgz.tar", "compressed.tar"];

fn path(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name);
    // The shards live at the workspace root, shared by every crate's tests, so
    // they are not part of the published package. Say so rather than letting a
    // bare "no such file" surface from inside the reader.
    assert!(path.exists(), "fixture {} not found; these tests run from a checkout of the repository", path.display());
    path.to_string_lossy().into_owned()
}

fn bytes(name: &str) -> Vec<u8> {
    std::fs::read(path(name)).expect("reading the shard")
}

fn blocking(name: &str) -> Vec<Sample> {
    WebDataset::builder_verbatim([path(name)]).build().expect("building").iter().map(|s| s.expect("a sample")).collect()
}

fn asynchronous(name: &str) -> Vec<Sample> {
    let dataset = AsyncWebDataset::builder_verbatim([path(name)]).build().expect("building");
    block_on(dataset.stream().try_collect::<Vec<_>>()).expect("no failures")
}

#[test]
fn reads_the_same_samples_from_every_shard() {
    for name in SHARDS {
        let expected = blocking(name);
        assert!(!expected.is_empty(), "{name} should not be empty");
        assert_eq!(asynchronous(name), expected, "{name} read differently");
    }
}

#[test]
fn decodes_the_same_values_from_every_shard() {
    for name in SHARDS {
        // `.mp` needs the msgpack decoder; without it both sides would agree
        // only on the error, which proves nothing about decoding.
        if *name == "mpdata.tar" && cfg!(not(feature = "msgpack")) {
            continue;
        }
        let expected: Vec<Sample> = WebDataset::builder_verbatim([path(name)])
            .build()
            .expect("building")
            .decode(Decoder::default())
            .iter()
            .map(|s| s.expect("a sample"))
            .collect();

        let dataset =
            AsyncWebDataset::builder_verbatim([path(name)]).build().expect("building").decode(Decoder::default());
        let got = block_on(dataset.stream().try_collect::<Vec<_>>()).expect("no failures");

        assert_eq!(got, expected, "{name} decoded differently");
    }
}

#[test]
fn batches_identically() {
    let name = "imagenet-000000.tgz";

    let expected: Vec<Sample> = WebDataset::builder_verbatim([path(name)])
        .build()
        .expect("building")
        .decode_basic()
        .iter()
        .batched(16, true)
        .map(|b| b.expect("a batch"))
        .collect();

    let dataset = AsyncWebDataset::builder_verbatim([path(name)]).build().expect("building").decode_basic();
    let got: Vec<Sample> = block_on(dataset.stream().batched(16, true).try_collect()).expect("no failures");

    assert_eq!(got.len(), 3);
    assert_eq!(got, expected);
}

#[test]
fn projects_to_tuples_identically() {
    let name = "imagenet-000000.tgz";

    let expected: Vec<Vec<Value>> = WebDataset::builder_verbatim([path(name)])
        .build()
        .expect("building")
        .decode_basic()
        .iter()
        .to_tuple(["png", "cls"])
        .batched(8, true)
        .map(|r| r.expect("a batch"))
        .collect();

    let dataset = AsyncWebDataset::builder_verbatim([path(name)]).build().expect("building").decode_basic();
    let got: Vec<Vec<Value>> =
        block_on(dataset.stream().to_tuple(["png", "cls"]).batched(8, true).try_collect()).expect("no failures");

    assert_eq!(got, expected);
}

#[test]
fn shuffles_identically_for_a_fixed_seed() {
    let name = "imagenet-000000.tgz";

    let expected: Vec<String> = WebDataset::builder_verbatim([path(name)])
        .seed(1234)
        .build()
        .expect("building")
        .shuffle(32)
        .iter()
        .map(|s| s.expect("a sample").key().expect("a key").to_string())
        .collect();

    let dataset = AsyncWebDataset::builder_verbatim([path(name)]).seed(1234).build().expect("building").shuffle(32);
    let got: Vec<String> = block_on(dataset.stream().try_collect::<Vec<_>>())
        .expect("no failures")
        .into_iter()
        .map(|s| s.key().expect("a key").to_string())
        .collect();

    assert_eq!(got, expected, "the same seed must give the same order either way");
}

#[test]
fn reads_shards_held_in_memory() {
    let opener = MemoryOpener::new().with("mem://a.tar", bytes("sample.tgz"));
    let dataset =
        AsyncWebDataset::builder_verbatim(["mem://a.tar"]).opener(Arc::new(opener)).build().expect("building");

    let samples = block_on(dataset.stream().try_collect::<Vec<_>>()).expect("no failures");
    assert_eq!(samples.len(), 90);
    assert_eq!(samples[0].url(), Some("mem://a.tar"));
}

#[test]
fn fetching_shards_concurrently_keeps_the_whole_sample_set() {
    let opener = || {
        MemoryOpener::new()
            .with("mem://a.tar", bytes("sample.tgz"))
            .with("mem://b.tar", bytes("imagenet-000000.tgz"))
            .with("mem://c.tar", bytes("mpdata.tar"))
            .with("mem://d.tar", bytes("tendata.tar"))
    };
    let urls = ["mem://a.tar", "mem://b.tar", "mem://c.tar", "mem://d.tar"];

    let keys = |concurrency: usize| -> Vec<String> {
        let dataset = AsyncWebDataset::builder_verbatim(urls)
            .opener(Arc::new(opener()))
            .concurrency(concurrency)
            .build()
            .expect("building");
        let mut keys: Vec<String> = block_on(dataset.stream().try_collect::<Vec<_>>())
            .expect("no failures")
            .into_iter()
            .map(|s| s.key().expect("a key").to_string())
            .collect();
        keys.sort();
        keys
    };

    let sequential = keys(1);
    assert_eq!(sequential.len(), 90 + 47 + 100 + 100);
    assert_eq!(keys(4), sequential, "concurrency may reorder, never lose or duplicate");
}

#[test]
fn an_unreadable_shard_is_reported() {
    let dataset =
        AsyncWebDataset::builder_verbatim(["/no/such/shard.tar"]).empty_check(false).build().expect("building");
    let outcome = block_on(dataset.stream().collect::<Vec<_>>());
    assert!(outcome.iter().any(Result::is_err));
}

#[test]
fn an_unreadable_shard_can_be_skipped() {
    let dataset = AsyncWebDataset::builder_verbatim(["/no/such/shard.tar".to_string(), path("sample.tgz")])
        .handler(webdataset::handlers::ignore_and_continue())
        .build()
        .expect("building");
    let samples = block_on(dataset.stream().try_collect::<Vec<_>>()).expect("the bad shard is skipped");
    assert_eq!(samples.len(), 90);
}

#[cfg(feature = "image")]
#[test]
fn decodes_images_identically() {
    let name = "imagenet-000000.tgz";

    let expected: Vec<Sample> = WebDataset::builder_verbatim([path(name)])
        .build()
        .expect("building")
        .decode_images("rgb8")
        .expect("a valid imagespec")
        .iter()
        .map(|s| s.expect("a sample"))
        .collect();

    let dataset = AsyncWebDataset::builder_verbatim([path(name)])
        .build()
        .expect("building")
        .decode_images("rgb8")
        .expect("a valid imagespec");
    let got = block_on(dataset.stream().try_collect::<Vec<_>>()).expect("no failures");

    assert_eq!(got, expected);
    assert_eq!(got[0].get("png").and_then(Value::as_tensor).expect("an image").shape(), &[793, 600, 3]);
}
