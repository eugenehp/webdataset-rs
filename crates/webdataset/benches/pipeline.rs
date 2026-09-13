//! Benchmarks for the parts of the pipeline that do the work.
//!
//! Shards are built in memory rather than read from `testdata/`, so the numbers
//! measure this library rather than the disk, and so the benchmarks run
//! anywhere without fixtures.
//!
//! ```console
//! $ cargo bench -p webdataset
//! $ cargo bench -p webdataset -- decode      # just the decoders
//! ```
//!
//! The comparison against the reference Python implementation lives in
//! `bench/`; this is for catching regressions in the Rust code itself.

// `criterion_group!` expands to a function without a doc comment, and the
// workspace warns on those. A benchmark harness is not public API.
#![allow(missing_docs)]

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use webdataset::core::Tensor;
use webdataset::filters::{SampleIteratorExt, TupleIteratorExt};
use webdataset::pipeline::{DataPipeline, Samples};
use webdataset::sources::{MemoryOpener, ShardsToSamples};
use webdataset::stages::{Decode, Shuffle};
use webdataset::{DefaultEncoder, Sample, SimpleShardList, TarWriter, Value};

/// How many samples each synthetic shard holds.
const PER_SHARD: usize = 512;

/// Build a shard in memory, with the fields a real one would have.
///
/// The payload is pseudo-random but deterministic, so runs are comparable.
fn make_shard(samples: usize, payload: usize) -> Vec<u8> {
    let buffer = SharedBuffer::default();
    let mut writer = TarWriter::new(Box::new(buffer.clone()), webdataset::shard::Compression::None)
        .expect("a tar writer over a Vec cannot fail")
        .with_encoder(Arc::new(DefaultEncoder::new()))
        .with_mtime(Some(0));

    let mut state = 0x2545_f491_4f6c_dd1du64;
    for i in 0..samples {
        let mut blob = Vec::with_capacity(payload);
        for _ in 0..payload {
            // xorshift: cheap, deterministic, and not compressible.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            blob.push(state as u8);
        }
        let mut sample = Sample::with_key(format!("sample{i:06}"));
        sample.insert("bin", Value::Bytes(blob.into()));
        sample.insert("cls", Value::Int((i % 1000) as i64));
        sample.insert("txt", Value::Text(format!("sample number {i}")));
        sample.insert("json", Value::from(serde_json::json!({ "index": i, "even": i % 2 == 0 })));
        sample.insert("npy", Value::Tensor(Tensor::from_f32(&[i as f32, 0.5, 1.5, 2.5])));
        writer.write(&sample).expect("writing to a Vec cannot fail");
    }
    writer.close().expect("closing a Vec cannot fail");
    buffer.take()
}

/// A `Vec<u8>` a boxed `'static` writer can hand back.
#[derive(Clone, Default)]
struct SharedBuffer(Arc<std::sync::Mutex<Vec<u8>>>);

impl SharedBuffer {
    fn take(self) -> Vec<u8> {
        std::mem::take(&mut *self.0.lock().expect("buffer lock"))
    }
}

impl std::io::Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("buffer lock").extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A pipeline over one in-memory shard.
fn pipeline_over(bytes: Vec<u8>) -> DataPipeline {
    let opener = MemoryOpener::new().with("mem://shard.tar", bytes);
    DataPipeline::new()
        .with(SimpleShardList::verbatim(["mem://shard.tar"]))
        .with(ShardsToSamples::new(Arc::new(opener)))
}

/// Reading a shard: archive parsing and grouping, with nothing decoded.
fn read(c: &mut Criterion) {
    let mut group = c.benchmark_group("read");
    for payload in [64usize, 4096, 65536] {
        let shard = make_shard(PER_SHARD, payload);
        group.throughput(Throughput::Bytes(shard.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(payload), &shard, |b, shard| {
            b.iter(|| {
                let pipeline = pipeline_over(shard.clone());
                black_box(pipeline.iter().filter(Result::is_ok).count())
            });
        });
    }
    group.finish();
}

/// Decoding: the same samples, put through the default handler chain.
fn decode(c: &mut Criterion) {
    let shard = make_shard(PER_SHARD, 256);
    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(PER_SHARD as u64));

    group.bench_function("none", |b| {
        b.iter(|| black_box(pipeline_over(shard.clone()).iter().filter(Result::is_ok).count()))
    });
    group.bench_function("basic", |b| {
        b.iter(|| {
            let pipeline = pipeline_over(shard.clone()).with(Decode::basic());
            black_box(pipeline.iter().filter(Result::is_ok).count())
        })
    });
    group.finish();
}

/// Field-level decoders, measured without the archive around them.
fn decoders(c: &mut Criterion) {
    let decoder = webdataset::Decoder::default();
    let tensor = Tensor::from_f32(&(0..1024).map(|i| i as f32).collect::<Vec<_>>());
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("cls", b"427".to_vec()),
        ("txt", "a caption of roughly the length one might expect".as_bytes().to_vec()),
        ("json", serde_json::to_vec(&serde_json::json!({"a": 1, "b": [1, 2, 3], "c": "x"})).expect("json")),
        ("npy", webdataset::core::npy::to_npy(&tensor)),
        ("ten", webdataset::tenbin::encode_buffer(std::slice::from_ref(&tensor))),
    ];

    let mut group = c.benchmark_group("decoders");
    for (field, data) in &cases {
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(field), data, |b, data| {
            b.iter(|| black_box(decoder.decode_field(field, data).expect("decodes")))
        });
    }
    group.finish();
}

/// Shuffling, which is pure buffer management over already-read samples.
fn shuffle(c: &mut Criterion) {
    let samples: Vec<Sample> = (0..4096)
        .map(|i| {
            let mut s = Sample::with_key(format!("k{i}"));
            s.insert("cls", Value::Int(i));
            s
        })
        .collect();

    let mut group = c.benchmark_group("shuffle");
    group.throughput(Throughput::Elements(samples.len() as u64));
    for bufsize in [100usize, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(bufsize), &bufsize, |b, &bufsize| {
            b.iter(|| {
                let pipeline =
                    DataPipeline::new().with(Samples::new(samples.clone())).with(Shuffle::new(bufsize).with_seed(1));
                black_box(pipeline.iter().filter(Result::is_ok).count())
            })
        });
    }
    group.finish();
}

/// Batching, which is collation: stacking columns into tensors.
fn batch(c: &mut Criterion) {
    let samples: Vec<Sample> = (0..2048)
        .map(|i| {
            let mut s = Sample::with_key(format!("k{i}"));
            s.insert("cls", Value::Int(i));
            s.insert("vec", Value::Tensor(Tensor::from_f32(&[i as f32; 64])));
            s
        })
        .collect();

    let mut group = c.benchmark_group("batch");
    group.throughput(Throughput::Elements(samples.len() as u64));
    for size in [16usize, 64, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let stream = samples.clone().into_iter().map(Ok);
                black_box(stream.batched(size, true).filter(Result::is_ok).count())
            })
        });
    }
    group.finish();
}

/// A whole training-shaped pipeline, end to end.
fn end_to_end(c: &mut Criterion) {
    let shard = make_shard(PER_SHARD, 1024);
    let mut group = c.benchmark_group("end_to_end");
    group.throughput(Throughput::Elements(PER_SHARD as u64));
    group.bench_function("shuffle+decode+tuple+batch", |b| {
        b.iter(|| {
            let pipeline = pipeline_over(shard.clone()).with(Shuffle::new(1000).with_seed(1)).with(Decode::basic());
            let batches = pipeline.iter().to_tuple(["bin", "cls"]).batched(64, true);
            black_box(batches.filter(Result::is_ok).count())
        })
    });
    group.finish();
}

/// Writing shards, the other direction.
fn write(c: &mut Criterion) {
    let mut group = c.benchmark_group("write");
    group.throughput(Throughput::Elements(PER_SHARD as u64));
    group.bench_function("512 samples", |b| b.iter(|| black_box(make_shard(PER_SHARD, 1024).len())));
    group.finish();
}

criterion_group!(benches, read, decode, decoders, shuffle, batch, end_to_end, write);
criterion_main!(benches);
