//! Tests against the shards shipped with the reference Python implementation.
//!
//! The expected values here are taken from that project's test suite, so a
//! passing run means this port reads the same samples, with the same keys,
//! fields, shapes and values, as the implementation it is a port of.

use std::path::PathBuf;

use webdataset::filters::{SampleIteratorExt, TupleIteratorExt};
use webdataset::{Decoder, Sample, Value, WebDataset};

/// Path to one of the bundled shards.
fn shard(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name);
    // The shards live at the workspace root, shared by every crate's tests, so
    // they are not part of the published package. Say so rather than letting a
    // bare "no such file" surface from inside the reader.
    assert!(path.exists(), "fixture {} not found; these tests run from a checkout of the repository", path.display());
    path.to_string_lossy().into_owned()
}

fn dataset(name: &str) -> WebDataset {
    WebDataset::builder_verbatim([shard(name)]).build().expect("building the dataset")
}

fn samples(name: &str) -> Vec<Sample> {
    dataset(name).iter().map(|s| s.expect("reading a sample")).collect()
}

#[test]
fn imagenet_shard_has_47_samples() {
    let samples = samples("imagenet-000000.tgz");
    assert_eq!(samples.len(), 47);

    let first = &samples[0];
    assert_eq!(first.key(), Some("10"));
    assert!(first.url().is_some());

    let mut fields = first.field_names();
    fields.sort();
    assert_eq!(fields, ["cls", "png", "wnid", "xml"]);
}

#[test]
fn imagenet_classes_decode_to_integers() {
    let samples: Vec<Sample> = dataset("imagenet-000000.tgz").decode_basic().iter().map(|s| s.unwrap()).collect();

    assert_eq!(samples.len(), 47);
    for sample in &samples {
        let cls = sample.get("cls").expect("cls").as_i64().expect("cls should decode to an integer");
        assert!(cls >= 0, "{cls}");
    }
}

#[test]
fn the_separator_concatenates_shard_lists() {
    // `a::b` is the reference implementation's way of naming two sources.
    let spec = format!("{}::{}", shard("imagenet-000000.tgz"), shard("imagenet-000000.tgz"));
    let dataset = WebDataset::builder(spec).build().unwrap();
    assert_eq!(dataset.iter().count(), 47 * 2);
}

#[test]
fn brace_patterns_expand_to_one_shard_per_index() {
    let shards = webdataset::SimpleShardList::new(["test-{000000..000099}.tar"]).unwrap();
    assert_eq!(shards.urls().len(), 100);
    assert_eq!(shards.urls()[0], "test-000000.tar");
    assert_eq!(shards.urls()[99], "test-000099.tar");
}

#[test]
fn tenbin_shard_holds_two_28x28_float64_arrays_per_sample() {
    let samples: Vec<Sample> = dataset("tendata.tar").decode_basic().iter().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 100);

    for sample in &samples {
        let tensors = sample.get("ten").expect("ten").as_list().expect("ten should decode to a list");
        assert_eq!(tensors.len(), 2);
        for tensor in tensors {
            let tensor = tensor.as_tensor().expect("a tensor");
            assert_eq!(tensor.dtype(), webdataset::DType::F64);
            assert_eq!(tensor.shape(), &[28, 28]);
        }
    }
}

#[test]
fn gzipped_fields_are_decompressed_then_decoded() {
    let sample = dataset("compressed.tar").decode_basic().iter().next().unwrap().unwrap();
    // The field keeps its `.gz` name but holds the decompressed, decoded text.
    assert_eq!(sample.get("txt.gz"), Some(&Value::Text("hello\n".into())));
    assert!(sample.url().is_some());
}

#[test]
fn a_shard_of_gzipped_text_decodes_throughout() {
    let samples: Vec<Sample> = dataset("testgz.tar").decode_basic().iter().map(|s| s.unwrap()).collect();
    assert!(!samples.is_empty());
    for sample in &samples {
        assert!(sample.get("txt.gz").and_then(Value::as_str).is_some(), "{:?}", sample.field_names());
    }
}

#[test]
fn samples_can_be_projected_to_tuples() {
    let rows: Vec<Vec<Value>> =
        dataset("imagenet-000000.tgz").decode_basic().iter().to_tuple(["png;jpg", "cls"]).map(|r| r.unwrap()).collect();

    assert_eq!(rows.len(), 47);
    assert_eq!(rows[0].len(), 2);
    assert!(rows[0][0].as_bytes().is_some(), "the image stays raw without an image handler");
    assert!(rows[0][1].as_i64().is_some());
}

#[test]
fn reading_the_same_shard_ten_times_gives_ten_epochs_worth() {
    let dataset = WebDataset::builder_verbatim(vec![shard("imagenet-000000.tgz"); 10]).build().unwrap();
    assert_eq!(dataset.iter().count(), 470);
}

#[test]
fn resampling_is_repeatable_across_epochs() {
    let dataset = WebDataset::builder_verbatim(vec![shard("imagenet-000000.tgz"); 3])
        .resampled(true)
        .build()
        .unwrap()
        .with_epoch(470);

    assert_eq!(dataset.iter().count(), 470);
    assert_eq!(dataset.iter().count(), 470, "a second epoch works just as well");
}

#[test]
fn selecting_files_drops_the_other_fields() {
    let dataset = WebDataset::builder_verbatim([shard("imagenet-000000.tgz")])
        .selection(webdataset::Selection::new().select(|name| name.ends_with(".png")))
        .build()
        .unwrap();

    let samples: Vec<Sample> = dataset.iter().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 47);
    assert_eq!(samples[0].field_names(), ["png"]);
    assert!(samples[0].key().is_some());
    assert!(samples[0].url().is_some());
}

#[test]
fn renaming_files_changes_the_field_names() {
    let dataset = WebDataset::builder_verbatim([shard("imagenet-000000.tgz")])
        .selection(webdataset::Selection::new().rename(|name| match name.strip_suffix(".cls") {
            Some(base) => format!("{base}.txt"),
            None => name.to_string(),
        }))
        .build()
        .unwrap();

    let sample = dataset.decode_basic().iter().next().unwrap().unwrap();
    assert!(!sample.contains_key("cls"));
    let text = sample.get("txt").and_then(Value::as_str).expect("txt should decode to text");
    assert!(text.trim().parse::<i64>().is_ok(), "the class is still a number: {text:?}");
}

#[test]
fn decoding_only_named_fields_leaves_the_rest_raw() {
    let decoder = Decoder::default().only(["cls"]);
    let sample = dataset("imagenet-000000.tgz").decode(decoder).iter().next().unwrap().unwrap();

    assert!(sample.get("cls").unwrap().as_i64().is_some());
    assert!(sample.get("png").unwrap().as_bytes().is_some());
}

#[test]
fn batching_stacks_the_class_column() {
    let batches: Vec<Vec<Value>> = dataset("imagenet-000000.tgz")
        .decode_basic()
        .iter()
        .to_tuple(["cls"])
        .batched(16, true)
        .map(|b| b.unwrap())
        .collect();

    assert_eq!(batches.len(), 3, "47 samples in batches of 16 leaves a partial batch");
    assert_eq!(batches[0][0].as_tensor().unwrap().shape(), &[16]);
    assert_eq!(batches[2][0].as_tensor().unwrap().shape(), &[15]);
}

#[test]
fn a_worker_pool_sees_each_sample_once() {
    let dataset =
        WebDataset::builder_verbatim(vec![shard("imagenet-000000.tgz"), shard("sample.tgz")]).build().unwrap();

    let mut keys: Vec<String> =
        dataset.loader().with_workers(2).iter().map(|s| s.unwrap().key().unwrap().to_string()).collect();
    keys.sort();

    let mut expected: Vec<String> = dataset.iter().map(|s| s.unwrap().key().unwrap().to_string()).collect();
    expected.sort();

    assert_eq!(keys, expected);
}

#[cfg(feature = "subprocess")]
#[test]
fn a_truncated_shard_is_reported_rather_than_silently_short() {
    // Reading only the first 10 kB of a shard leaves the archive truncated.
    let url = format!("pipe:dd if={} bs=1024 count=10 2>/dev/null", shard("imagenet-000000.tgz"));
    let dataset = WebDataset::builder_verbatim([url]).empty_check(false).build().unwrap();

    let outcome: Vec<_> = dataset.iter().collect();
    let good = outcome.iter().filter(|s| s.is_ok()).count();
    assert!(good < 47, "a truncated shard cannot hold every sample, got {good}");
    assert!(outcome.iter().any(Result::is_err), "the truncation should be reported");
}

#[cfg(feature = "subprocess")]
#[test]
fn a_truncated_shard_can_be_skipped_instead() {
    let url = format!("pipe:dd if={} bs=1024 count=10 2>/dev/null", shard("imagenet-000000.tgz"));
    let dataset = WebDataset::builder_verbatim([url])
        .handler(webdataset::handlers::ignore_and_continue())
        .empty_check(false)
        .build()
        .unwrap();

    let samples: Vec<Sample> = dataset.iter().map(|s| s.unwrap()).collect();
    assert!(!samples.is_empty(), "the samples before the truncation are still usable");
    assert!(samples.len() < 47);
}

#[cfg(feature = "msgpack")]
#[test]
fn messagepack_shard_decodes() {
    let samples: Vec<Sample> = dataset("mpdata.tar").decode_basic().iter().map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 100);
    assert!(samples[0].get("mp").is_some());
}

#[cfg(feature = "image")]
#[test]
fn imagenet_images_decode_to_793x600x3() {
    let dataset = dataset("imagenet-000000.tgz").decode_images("rgb").unwrap();
    let sample = dataset.iter().next().unwrap().unwrap();

    let image = sample.get("png").unwrap().as_tensor().expect("png should decode to a tensor");
    assert_eq!(image.shape(), &[793, 600, 3]);
    assert_eq!(image.dtype(), webdataset::DType::F32);
    assert!(image.to_f64_vec().iter().all(|v| (0.0..=1.0).contains(v)));
}

#[cfg(feature = "image")]
#[test]
fn the_torch_imagespec_transposes_the_same_pixels() {
    let hwc = dataset("imagenet-000000.tgz").decode_images("rgb8").unwrap().iter().next().unwrap().unwrap();
    let chw = dataset("imagenet-000000.tgz").decode_images("torchrgb8").unwrap().iter().next().unwrap().unwrap();

    let hwc = hwc.get("png").unwrap().as_tensor().unwrap().clone();
    let chw = chw.get("png").unwrap().as_tensor().unwrap().clone();

    assert_eq!(hwc.shape(), &[793, 600, 3]);
    assert_eq!(chw.shape(), &[3, 793, 600]);

    // Pixel (0, 0) channel 1 must be the same value under either layout.
    let (height, width) = (793usize, 600usize);
    assert_eq!(hwc.get(1), chw.get(height * width));
}
