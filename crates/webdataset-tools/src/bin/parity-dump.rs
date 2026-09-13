//! Dump a shard's contents in the same form as `parity/dump_python.py`.
//!
//! This is repository tooling rather than an example: it exists so that
//! `parity/run_parity.py` has something to compare the reference implementation
//! against, and it is not published.
//!
//! Each sample becomes one JSON line holding its key and, per field, a type tag
//! and a SHA-256 digest of a canonical byte form of the decoded value. Running
//! this and the Python dumper over the same shards and diffing the output is a
//! direct check that this port reads the format identically.
//!
//! ```sh
//! cargo run --release -p webdataset-tools --features libjpeg --bin parity-dump -- \
//!     testdata/sample.tgz --decode basic
//! ```

use std::collections::BTreeMap;
use std::io::Write;

use webdataset::{Decoder, Result, Sample, Value, WebDataset};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(shards) = args.next() else {
        eprintln!("usage: parity_dump <shards> [--decode none|basic|<imagespec>] [--limit N]");
        return Ok(());
    };

    let mut decode = "none".to_string();
    let mut limit = 0usize;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--decode" => decode = args.next().unwrap_or_else(|| "none".into()),
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            other => {
                eprintln!("unknown flag {other}");
                return Ok(());
            }
        }
    }

    let mut dataset = WebDataset::builder(&shards).empty_check(false).build()?;
    dataset = match decode.as_str() {
        "none" => dataset,
        "basic" => dataset.decode(Decoder::default()),
        spec => dataset.decode_images(spec)?,
    };
    if limit > 0 {
        dataset = dataset.take(limit);
    }

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for sample in dataset.iter() {
        writeln!(out, "{}", line(&sample?))?;
    }
    out.flush()?;
    Ok(())
}

/// One JSON line describing a sample, with fields in sorted order.
fn line(sample: &Sample) -> String {
    let fields: BTreeMap<&str, (String, String)> = sample
        .iter()
        .filter(|(name, _)| !name.starts_with("__"))
        .map(|(name, value)| (name.as_str(), (describe(value), digest(value))))
        .collect();

    let rendered: Vec<String> = fields
        .iter()
        .map(|(name, (kind, sha))| format!("{}: {{\"sha256\": \"{sha}\", \"type\": \"{kind}\"}}", quote(name)))
        .collect();

    format!("{{\"fields\": {{{}}}, \"key\": {}}}", rendered.join(", "), quote(sample.key().unwrap_or("")))
}

/// A short type tag, matching the Python dumper's vocabulary.
fn describe(value: &Value) -> String {
    match value {
        Value::Bytes(b) => format!("bytes[{}]", b.len()),
        Value::Text(_) => "str".to_string(),
        Value::Int(_) => "int".to_string(),
        Value::Float(_) => "float".to_string(),
        Value::Bool(_) => "bool".to_string(),
        Value::Null => "NoneType".to_string(),
        Value::List(_) => "list".to_string(),
        Value::Map(_) => "dict".to_string(),
        Value::Tensor(t) => format!("tensor<{}>{:?}", t.dtype().long_name(), t.shape()),
        Value::Image(_) => {
            let (shape, _) = image_array(value);
            format!("tensor<uint8>{shape:?}")
        }
        other => other.type_name().to_string(),
    }
}

/// A decoded image as the shape and bytes NumPy would see.
///
/// `np.asarray` on a PIL image gives height × width × channels, with the
/// channel axis dropped for a single-channel image, so a decoded image compares
/// equal to the tensor the numeric imagespecs produce.
fn image_array(value: &Value) -> (Vec<usize>, Vec<u8>) {
    use image::DynamicImage;

    let Some(image) = value.as_image() else {
        return (Vec::new(), Vec::new());
    };
    let (width, height) = (image.width() as usize, image.height() as usize);
    match image {
        DynamicImage::ImageLuma8(buffer) => (vec![height, width], buffer.as_raw().clone()),
        DynamicImage::ImageRgba8(buffer) => (vec![height, width, 4], buffer.as_raw().clone()),
        other => (vec![height, width, 3], other.to_rgb8().into_raw()),
    }
}

/// The SHA-256 of the canonical byte form, so only decoded data is compared.
fn digest(value: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(&canonical(value));
    hasher.finish()
}

/// The canonical byte form; see `parity/dump_python.py` for the other half.
fn canonical(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    match value {
        Value::Bytes(b) => {
            out.extend_from_slice(b"bytes:");
            out.extend_from_slice(b);
        }
        Value::Text(s) => {
            out.extend_from_slice(b"text:");
            out.extend_from_slice(s.as_bytes());
        }
        Value::Bool(b) => {
            out.extend_from_slice(b"bool:");
            out.extend_from_slice(if *b { b"1" } else { b"0" });
        }
        Value::Int(i) => {
            out.extend_from_slice(b"int:");
            out.extend_from_slice(i.to_string().as_bytes());
        }
        Value::Float(f) => {
            out.extend_from_slice(b"float:");
            out.extend_from_slice(python_repr(*f).as_bytes());
        }
        Value::Null => out.extend_from_slice(b"null"),
        Value::List(items) => {
            out.extend_from_slice(b"list:[");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend_from_slice(&canonical(item));
            }
            out.push(b']');
        }
        Value::Map(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.extend_from_slice(b"map:{");
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend_from_slice(key.as_bytes());
                out.push(b'=');
                out.extend_from_slice(&canonical(&map[key.as_str()]));
            }
            out.push(b'}');
        }
        Value::Tensor(t) => {
            out.extend_from_slice(format!("tensor:{}:{:?}:", t.dtype().long_name(), t.shape()).as_bytes());
            out.extend_from_slice(t.data());
        }
        Value::Image(_) => {
            let (shape, bytes) = image_array(value);
            out.extend_from_slice(format!("tensor:uint8:{shape:?}:").as_bytes());
            out.extend_from_slice(&bytes);
        }
        other => out.extend_from_slice(format!("unsupported:{}", other.type_name()).as_bytes()),
    }
    out
}

/// Format a float the way Python's `repr` does, so digests agree.
fn python_repr(value: f64) -> String {
    if value == value.trunc() && value.abs() < 1e16 {
        return format!("{value:.1}");
    }
    let mut shortest = format!("{value}");
    for precision in 1..=17 {
        let candidate = format!("{value:.precision$e}");
        if candidate.parse::<f64>() == Ok(value) {
            shortest = format!("{value}");
            break;
        }
    }
    shortest
}

/// JSON-quote a string.
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A self-contained SHA-256, so the example needs no extra dependency.
struct Sha256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    length: u64,
}

impl Sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    fn new() -> Sha256 {
        Sha256 {
            state: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19],
            buffer: Vec::with_capacity(64),
            length: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.length += data.len() as u64;
        let mut data = data;

        // Top up the pending partial block first.
        if !self.buffer.is_empty() {
            let wanted = (64 - self.buffer.len()).min(data.len());
            self.buffer.extend_from_slice(&data[..wanted]);
            data = &data[wanted..];
            if self.buffer.len() == 64 {
                let block: [u8; 64] = self.buffer[..].try_into().expect("64 bytes");
                self.compress(&block);
                self.buffer.clear();
            }
        }

        // Then take whole blocks straight from the input. Buffering the input
        // and draining 64 bytes at a time would shift the remainder on every
        // block, which is quadratic and unusable on a multi-megabyte image.
        let mut blocks = data.chunks_exact(64);
        for block in &mut blocks {
            self.compress(block.try_into().expect("64 bytes"));
        }
        self.buffer.extend_from_slice(blocks.remainder());
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            *word = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for (k, word) in Self::K.iter().zip(w.iter()) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(*k).wrapping_add(*word);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    fn finish(mut self) -> String {
        let bits = self.length * 8;
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0);
        }
        let tail = bits.to_be_bytes();
        self.buffer.extend_from_slice(&tail);
        // At most two blocks remain here, so copying them out to satisfy the
        // borrow checker costs nothing.
        let blocks: Vec<[u8; 64]> = self.buffer.chunks_exact(64).map(|b| b.try_into().expect("64 bytes")).collect();
        for block in blocks {
            self.compress(&block);
        }
        self.state.iter().map(|word| format!("{word:08x}")).collect()
    }
}
