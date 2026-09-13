//! Writing a dataset and reading it back.

use std::sync::Arc;

use webdataset::filters::SampleIteratorExt;
use webdataset::{DefaultEncoder, Sample, ShardWriter, Value, WebDataset};
use webdataset_core::Tensor;

/// Build a sample with one field of each interesting kind.
fn sample(i: i64) -> Sample {
    let mut sample = Sample::with_key(format!("sample{i:06}"));
    sample.insert("cls", Value::Int(i % 10));
    sample.insert("txt", Value::Text(format!("this is sample number {i}")));
    sample.insert("json", Value::from(serde_json::json!({ "index": i, "even": i % 2 == 0 })));
    sample.insert("npy", Value::Tensor(Tensor::from_f32(&[i as f32, (i * 2) as f32])));
    sample
}

#[test]
fn a_dataset_survives_a_write_and_read_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let pattern = dir.path().join("train-%06d.tar");

    let mut writer = ShardWriter::new(pattern.to_str().unwrap())
        .unwrap()
        .with_encoder(Arc::new(DefaultEncoder::new()))
        .with_max_count(40);
    for i in 0..100 {
        writer.write(&sample(i)).unwrap();
    }
    assert_eq!(writer.total(), 100);
    writer.close().unwrap();

    let shards: Vec<String> =
        (0..3).map(|i| dir.path().join(format!("train-{i:06}.tar")).to_string_lossy().into_owned()).collect();
    assert!(shards.iter().all(|s| std::path::Path::new(s).exists()), "three shards should have been written");

    let dataset = WebDataset::builder_verbatim(shards).build().unwrap().decode_basic();
    let read: Vec<Sample> = dataset.iter().map(|s| s.unwrap()).collect();

    assert_eq!(read.len(), 100);
    for (i, sample) in read.iter().enumerate() {
        let i = i as i64;
        assert_eq!(sample.key(), Some(format!("sample{i:06}").as_str()));
        assert_eq!(sample.get("cls").unwrap().as_i64(), Some(i % 10));
        assert_eq!(sample.get("txt").unwrap().as_str(), Some(format!("this is sample number {i}").as_str()));
        assert_eq!(sample.get("npy").unwrap().as_tensor().unwrap(), &Tensor::from_f32(&[i as f32, (i * 2) as f32]));

        let json = sample.get("json").unwrap().as_map().unwrap();
        assert_eq!(json.get("index").unwrap().as_i64(), Some(i));
        assert_eq!(json.get("even"), Some(&Value::Bool(i % 2 == 0)));
    }
}

#[test]
fn compressed_shards_round_trip_too() {
    let dir = tempfile::tempdir().unwrap();
    let pattern = dir.path().join("train-%d.tar.gz");

    let mut writer = ShardWriter::new(pattern.to_str().unwrap()).unwrap().with_encoder(Arc::new(DefaultEncoder::new()));
    for i in 0..20 {
        writer.write(&sample(i)).unwrap();
    }
    writer.close().unwrap();

    let path = dir.path().join("train-0.tar.gz");
    assert_eq!(&std::fs::read(&path).unwrap()[..2], &[0x1f, 0x8b], "the shard should be gzipped");

    let dataset = WebDataset::builder_verbatim([path.to_string_lossy().into_owned()]).build().unwrap();
    assert_eq!(dataset.iter().count(), 20);
}

#[test]
fn shuffling_preserves_the_sample_set() {
    let dir = tempfile::tempdir().unwrap();
    let pattern = dir.path().join("s-%d.tar");

    let mut writer = ShardWriter::new(pattern.to_str().unwrap())
        .unwrap()
        .with_encoder(Arc::new(DefaultEncoder::new()))
        .with_max_count(25);
    for i in 0..100 {
        writer.write(&sample(i)).unwrap();
    }
    writer.close().unwrap();

    let shards = dir.path().join("s-{0..3}.tar").to_string_lossy().into_owned();
    let dataset = WebDataset::builder(shards).shard_shuffle(4).seed(11).build().unwrap().shuffle(50);

    let mut keys: Vec<String> = dataset.iter().map(|s| s.unwrap().key().unwrap().to_string()).collect();
    assert_eq!(keys.len(), 100);

    let ordered: Vec<String> = (0..100).map(|i| format!("sample{i:06}")).collect();
    assert_ne!(keys, ordered, "shuffling should change the order");
    keys.sort();
    assert_eq!(keys, ordered, "but not the contents");
}

#[test]
fn a_batched_pipeline_produces_stacked_columns() {
    let dir = tempfile::tempdir().unwrap();
    let pattern = dir.path().join("s-%d.tar");

    let mut writer = ShardWriter::new(pattern.to_str().unwrap()).unwrap().with_encoder(Arc::new(DefaultEncoder::new()));
    for i in 0..64 {
        writer.write(&sample(i)).unwrap();
    }
    writer.close().unwrap();

    let shard = dir.path().join("s-0.tar").to_string_lossy().into_owned();
    let dataset = WebDataset::builder_verbatim([shard]).build().unwrap().decode_basic();

    let batches: Vec<Sample> = dataset.iter().batched(16, true).map(|b| b.unwrap()).collect();
    assert_eq!(batches.len(), 4);

    let first = &batches[0];
    assert_eq!(first.get("cls").unwrap().as_tensor().unwrap().shape(), &[16]);
    assert_eq!(first.get("npy").unwrap().as_tensor().unwrap().shape(), &[16, 2]);
    assert_eq!(first.get("txt").unwrap().as_list().unwrap().len(), 16);
    assert_eq!(first.get("__key__").unwrap().as_list().unwrap().len(), 16);
}

#[cfg(feature = "subprocess")]
#[test]
fn writing_to_a_pipe_url_works() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("piped.tar");

    let url = format!("pipe:cat > {}", path.display());
    let mut writer = webdataset::TarWriter::create(&url).unwrap().with_encoder(Arc::new(DefaultEncoder::new()));
    writer.write(&sample(1)).unwrap();
    writer.close().unwrap();

    let dataset = WebDataset::builder_verbatim([path.to_string_lossy().into_owned()]).build().unwrap();
    assert_eq!(dataset.iter().count(), 1);
}
