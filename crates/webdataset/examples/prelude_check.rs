//! Everything this example needs comes from the prelude and nothing else.
//!
//! It exists to keep that claim true: if the prelude stops carrying a type or
//! a trait the common path needs, this stops compiling. `cargo test` builds
//! the examples, so the check runs with the suite rather than on request.

use webdataset::prelude::*;

fn main() -> Result<()> {
    let shard = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/sample.tgz");

    // Types named from the prelude: WebDataset, Selection, HandlerRef.
    let dataset = WebDataset::builder(shard)
        .selection(Selection::new())
        .handler(handlers::reraise_exception())
        .seed(0)
        .build()?;

    // `.shuffled(...)` is a trait method: without SampleIteratorExt in scope
    // it does not exist. The trait is bounded on `Sized`, so the boxed stream
    // a dataset yields needs unboxing first.
    let stream = dataset.into_iter();
    let samples: Vec<Sample> = stream.shuffled(16, Some(0)).take(3).collect::<Result<Vec<_>>>()?;

    for sample in &samples {
        for (name, value) in sample.iter() {
            // `Value` and its accessors come from the prelude too.
            let bytes = value.as_bytes().map(|b| b.len()).unwrap_or(0);
            let _ = (name, bytes);
        }
    }

    let first = samples[0].key().unwrap_or("<no key>");
    println!("prelude ok: {} samples, first key {first}", samples.len());
    Ok(())
}
