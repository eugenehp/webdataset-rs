//! The pipeline shape that works everywhere, including WebAssembly.
//!
//! These tests avoid threads, subprocesses and the filesystem in the pipeline
//! itself: shard bytes are handed over in memory and everything downstream is
//! pure computation. Running them with `--no-default-features` exercises the
//! same code paths a `wasm32-unknown-unknown` build takes.

use std::sync::Arc;

use webdataset::filters::{SampleIteratorExt, TupleIteratorExt};
use webdataset::pipeline::DataPipeline;
use webdataset::shardlists::SimpleShardList;
use webdataset::sources::{MemoryOpener, ShardsToSamples};
use webdataset::stages::{Decode, Shuffle};
use webdataset::{DataLoader, Sample, Value};

/// The bytes of a bundled shard, as a host would hand them to a wasm module.
fn shard_bytes(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name);
    // The shards live at the workspace root, shared by every crate's tests, so
    // they are not part of the published package. Say so rather than letting a
    // bare "no such file" surface from inside the reader.
    assert!(path.exists(), "fixture {} not found; these tests run from a checkout of the repository", path.display());
    std::fs::read(path).expect("reading the test shard")
}

/// A pipeline over shards held in memory.
fn in_memory(name: &str) -> DataPipeline {
    let opener = MemoryOpener::new().with("mem://shard.tar", shard_bytes(name));
    DataPipeline::new()
        .with(SimpleShardList::verbatim(["mem://shard.tar"]))
        .with(ShardsToSamples::new(Arc::new(opener)))
}

#[test]
fn reads_a_gzipped_shard_from_memory() {
    let samples: Vec<Sample> = in_memory("imagenet-000000.tgz").iter().map(|s| s.unwrap()).collect();

    assert_eq!(samples.len(), 47);
    assert_eq!(samples[0].key(), Some("10"));
    assert!(samples[0].contains_key("png"));
    assert_eq!(samples[0].url(), Some("mem://shard.tar"));
}

#[test]
fn reads_an_uncompressed_shard_from_memory() {
    let samples: Vec<Sample> = in_memory("tendata.tar").iter().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 100);
}

#[test]
fn decodes_shuffles_and_batches_without_any_host_services() {
    let pipeline = in_memory("imagenet-000000.tgz").with(Shuffle::new(20).with_seed(7)).with(Decode::basic());

    let batches: Vec<Vec<Value>> = pipeline.iter().to_tuple(["cls"]).batched(16, true).map(|b| b.unwrap()).collect();

    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0][0].as_tensor().unwrap().shape(), &[16]);
    assert_eq!(batches[2][0].as_tensor().unwrap().shape(), &[15]);
}

#[test]
fn the_loader_still_works_with_a_single_worker() {
    // Without threads the loader runs inline whatever is asked for, so this
    // must produce the same samples either way.
    let pipeline = in_memory("sample.tgz");
    let direct = pipeline.iter().count();
    assert_eq!(DataLoader::new(pipeline.clone()).iter().count(), direct);
    assert_eq!(DataLoader::new(pipeline).with_workers(1).iter().count(), direct);
}

#[test]
fn epochs_can_be_replayed() {
    let pipeline = in_memory("sample.tgz").with_epoch(200);
    assert_eq!(pipeline.iter().count(), 200);
    assert_eq!(pipeline.iter().count(), 200);
}

#[test]
fn tenbin_and_npy_decode_without_the_filesystem() {
    let sample = in_memory("tendata.tar").with(Decode::basic()).iter().next().unwrap().unwrap();
    let tensors = sample.get("ten").unwrap().as_list().unwrap();
    assert_eq!(tensors.len(), 2);
    assert_eq!(tensors[0].as_tensor().unwrap().shape(), &[28, 28]);
}

#[test]
fn gzipped_fields_decode_without_the_filesystem() {
    let sample = in_memory("compressed.tar").with(Decode::basic()).iter().next().unwrap().unwrap();
    assert_eq!(sample.get("txt.gz"), Some(&Value::Text("hello\n".into())));
}

#[test]
fn shards_can_be_written_to_memory_and_read_back() {
    use webdataset::shard::{Compression, TarWriter};

    /// A `Vec<u8>` a boxed `'static` writer can hand back.
    #[derive(Clone, Default)]
    struct Shared(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let buffer = Shared::default();
    let mut writer = TarWriter::new(Box::new(buffer.clone()), Compression::Gzip)
        .unwrap()
        .with_encoder(Arc::new(webdataset::DefaultEncoder::new()))
        .with_mtime(Some(0));

    for i in 0..10i64 {
        let mut sample = Sample::with_key(format!("k{i}"));
        sample.insert("cls", Value::Int(i));
        sample.insert("txt", Value::Text(format!("sample {i}")));
        writer.write(&sample).unwrap();
    }
    writer.close().unwrap();

    let bytes = buffer.0.lock().expect("lock").clone();
    assert_eq!(&bytes[..2], &[0x1f, 0x8b], "the shard should be gzipped");

    let opener = MemoryOpener::new().with("mem://written.tar", bytes);
    let pipeline = DataPipeline::new()
        .with(SimpleShardList::verbatim(["mem://written.tar"]))
        .with(ShardsToSamples::new(Arc::new(opener)))
        .with(Decode::basic());

    let back: Vec<Sample> = pipeline.iter().map(|s| s.unwrap()).collect();
    assert_eq!(back.len(), 10);
    assert_eq!(back[3].get("cls").unwrap().as_i64(), Some(3));
    assert_eq!(back[3].get("txt").unwrap().as_str(), Some("sample 3"));
}
