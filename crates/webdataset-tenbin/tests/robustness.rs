//! Decoding must not panic on input it did not encode.
//!
//! `.ten` fields arrive inside shards, so the bytes are as untrusted as the
//! archive around them. The format is length-prefixed, which is the shape that
//! invites two mistakes: believing a length when allocating, and believing it
//! again when slicing. Both are checked here.

use webdataset_tenbin::{ALIGNMENT, MAGIC};

/// A deterministic bit source, so a failure can be reproduced from the seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

/// Try every decoder on the same bytes. Errors are expected; panics are not.
fn decode_every_way(bytes: &[u8]) {
    let _ = webdataset_tenbin::decode_buffer(bytes);
    let _ = webdataset_tenbin::decode_labelled(bytes);
    // The streaming readers need `std::io::Read`, so they only exist when the
    // crate is built with `std`. The slice decoders above cover the same
    // parsing either way.
    #[cfg(feature = "std")]
    {
        let mut cursor = std::io::Cursor::new(bytes);
        let _ = webdataset_tenbin::read(&mut cursor);
        let mut cursor = std::io::Cursor::new(bytes);
        let _ = webdataset_tenbin::read_chunk(&mut cursor);
    }
}

/// A chunk with the right magic and a declared length that need not be honest.
fn chunk(length: i64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&length.to_ne_bytes());
    out.extend_from_slice(payload);
    out
}

#[test]
fn a_length_larger_than_the_input_is_an_error_not_an_abort() {
    // The whole point: a chunk that claims more than any machine can allocate
    // must come back as an error rather than taking the process down.
    for length in [i64::MAX, i64::MAX / 2, 1 << 62, 1 << 48, 1 << 40, 1_000_000_000] {
        decode_every_way(&chunk(length, b"only a few bytes"));
    }
}

#[test]
fn a_negative_or_absurd_length_is_an_error_not_a_panic() {
    for length in [-1, i64::MIN, i64::MIN + 1, -64, -(ALIGNMENT as i64)] {
        decode_every_way(&chunk(length, b"payload"));
    }
}

#[test]
fn arbitrary_bytes_are_an_error_not_a_panic() {
    let mut rng = Rng(0x7E4B_1234);
    for _ in 0..2048 {
        let len = (rng.next() % 512) as usize;
        let bytes: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
        decode_every_way(&bytes);
    }
}

#[test]
fn bytes_behind_a_valid_magic_are_an_error_not_a_panic() {
    // Random bytes almost never start with the magic, so they never reach the
    // length handling. These do.
    let mut rng = Rng(0xBEEF_0001);
    for _ in 0..2048 {
        let mut bytes = Vec::from(*MAGIC);
        let len = (rng.next() % 256) as usize;
        bytes.extend((0..len).map(|_| rng.byte()));
        decode_every_way(&bytes);
    }
}

#[test]
fn a_truncated_encoding_is_an_error_not_a_panic() {
    let tensors = webdataset_tenbin::decode_buffer(&{
        let mut rng = Rng(1);
        let mut bytes = Vec::from(*MAGIC);
        bytes.extend((0..64).map(|_| rng.byte()));
        bytes
    });
    // Whether that parsed is beside the point; what follows needs real output.
    let _ = tensors;

    let whole = webdataset_tenbin::encode_buffer(&[webdataset_core::Tensor::new(
        webdataset_core::DType::F32,
        vec![2, 3],
        vec![0u8; 24],
    )
    .expect("a well formed tensor")]);

    for end in 0..whole.len() {
        decode_every_way(&whole[..end]);
    }
}
