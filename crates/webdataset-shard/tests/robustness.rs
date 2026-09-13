//! The parser must not panic on input it did not write.
//!
//! Shards arrive from elsewhere — downloaded, copied, truncated by a full disk,
//! or simply hostile. A hand-written parser that indexes into headers is where
//! that goes wrong, so this feeds it deliberately broken archives and requires
//! an error rather than a panic. Errors are the contract; panics are not.
//!
//! Random bytes alone are a weak test: almost all of them fail the header
//! checksum and never reach the code that reads a name or a size. The
//! interesting inputs are the ones that *look* valid — a correct checksum over
//! nonsense fields, which is also what a deliberately crafted archive would
//! carry. [`corrupt_header`] builds those.

use std::io::Cursor;

use webdataset_shard::TarFiles;

const BLOCK: usize = 512;

/// Read an archive to exhaustion, reporting how many entries came back.
///
/// Any return value is fine, including an error. The point is reaching this
/// line at all: a panic anywhere inside fails the test.
fn drain(bytes: Vec<u8>) -> usize {
    let Ok(files) = TarFiles::new(Box::new(Cursor::new(bytes))) else {
        return 0;
    };
    files.take(4096).filter(|entry| entry.is_ok()).count()
}

/// A deterministic bit source, so a failure can be reproduced from the seed
/// alone rather than from a saved corpus.
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

fn fixture() -> Vec<u8> {
    // The smallest fixture with several members. A bigger one would only
    // re-read the same header code more slowly.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/mpdata.tar");
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// The checksum a header block should carry: every byte summed, with the
/// checksum field itself counted as spaces.
fn checksum(block: &[u8; BLOCK]) -> u32 {
    let mut total: u32 = 0;
    for (index, byte) in block.iter().enumerate() {
        total += if (148..156).contains(&index) { b' ' as u32 } else { *byte as u32 };
    }
    total
}

/// A header block whose fields are arbitrary but whose checksum is correct.
///
/// This is the shape that gets past the cheap rejection and into the parsing,
/// so it is where a bounds mistake or an unchecked `unwrap` actually shows up.
fn corrupt_header(rng: &mut Rng) -> [u8; BLOCK] {
    let mut block = [0u8; BLOCK];

    // Plausible-looking fields, then damage. `ustar` in the magic keeps the
    // block from being dismissed as a V7 archive.
    block[..8].copy_from_slice(b"data.txt");
    block[100..108].copy_from_slice(b"000644 \0");
    block[124..136].copy_from_slice(b"00000000064\0");
    block[136..148].copy_from_slice(b"00000000000\0");
    block[156] = b'0';
    block[257..263].copy_from_slice(b"ustar\0");
    block[263..265].copy_from_slice(b"00");

    // Damage a handful of bytes anywhere except the checksum, which is
    // recomputed afterwards so the block still passes verification.
    for _ in 0..6 {
        let at = (rng.next() as usize) % BLOCK;
        if (148..156).contains(&at) {
            continue;
        }
        block[at] = rng.byte();
    }

    let sum = checksum(&block);
    let field = format!("{sum:06o}\0 ");
    block[148..156].copy_from_slice(field.as_bytes());
    block
}

#[test]
fn a_header_that_passes_its_checksum_but_holds_nonsense_is_an_error_not_a_panic() {
    let mut rng = Rng(0x1234_5678);
    for _ in 0..2048 {
        let mut bytes = corrupt_header(&mut rng).to_vec();
        // A payload, then the two zero blocks that end an archive.
        bytes.extend((0..BLOCK).map(|_| rng.byte()));
        bytes.extend([0u8; 2 * BLOCK]);
        drain(bytes);
    }
}

#[test]
fn a_size_field_larger_than_the_archive_is_an_error_not_a_panic() {
    let mut rng = Rng(0xFEED_BEEF);
    // Every width of octal size, including ones that overflow a u64, and the
    // GNU base-256 form that the top bit selects.
    for digits in 1..=12 {
        let mut block = corrupt_header(&mut rng);
        let claim: Vec<u8> = core::iter::repeat_n(b'7', digits).collect();
        block[124..136].fill(0);
        block[124..124 + digits].copy_from_slice(&claim);
        let sum = checksum(&block);
        block[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        drain(block.to_vec());
    }

    let mut block = corrupt_header(&mut rng);
    block[124] = 0x80; // GNU base-256 marker
    block[125..136].fill(0xFF);
    let sum = checksum(&block);
    block[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    drain(block.to_vec());
}

#[test]
fn every_typeflag_is_an_error_not_a_panic() {
    let mut rng = Rng(0xABCD);
    // `L` and `x` make the next block a name or a PAX record, which is extra
    // parsing on attacker-controlled bytes; the rest should be ignored.
    for flag in 0u8..=255 {
        let mut block = corrupt_header(&mut rng);
        block[156] = flag;
        let sum = checksum(&block);
        block[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());

        let mut bytes = block.to_vec();
        bytes.extend((0..2 * BLOCK).map(|_| rng.byte()));
        bytes.extend([0u8; 2 * BLOCK]);
        drain(bytes);
    }
}

#[test]
fn arbitrary_bytes_are_an_error_not_a_panic() {
    let mut rng = Rng(0x5EED);
    for _ in 0..512 {
        let len = (rng.next() % 4096) as usize;
        drain((0..len).map(|_| rng.byte()).collect());
    }
}

#[test]
fn a_shard_truncated_at_any_point_is_an_error_not_a_panic() {
    let whole = fixture();

    // Every boundary matters: mid-header, mid-name, mid-size field,
    // mid-payload. Reading a prefix costs time proportional to its length, so
    // walking all of them is quadratic; take a bounded sample instead. The
    // first two blocks are covered byte by byte, since that is where the header
    // parsing lives and where an off-by-one would hide.
    for end in 0..(2 * BLOCK).min(whole.len()) {
        drain(whole[..end].to_vec());
    }
    for end in (0..whole.len()).step_by(whole.len() / 256) {
        drain(whole[..end].to_vec());
    }
}

#[test]
fn a_shard_with_flipped_bytes_is_an_error_not_a_panic() {
    let whole = fixture();
    let mut rng = Rng(0xC0FFEE);

    for _ in 0..512 {
        let mut bytes = whole.clone();
        // Concentrate on the first few blocks, where the headers are: random
        // damage to a payload proves much less than damage to a size field.
        let reach = bytes.len().min(4096);
        for _ in 0..8 {
            let at = (rng.next() as usize) % reach;
            bytes[at] = rng.byte();
        }
        drain(bytes);
    }
}

/// The async reader parses headers with its own copy of the same logic, so it
/// needs the same guarantee rather than inheriting it.
#[cfg(feature = "async")]
#[test]
fn the_async_reader_is_an_error_not_a_panic_on_the_same_input() {
    use futures_executor::block_on;
    use webdataset_shard::asynch::AsyncTarEntries;

    let mut rng = Rng(0x1234_5678);
    block_on(async {
        for _ in 0..512 {
            let mut bytes = corrupt_header(&mut rng).to_vec();
            bytes.extend((0..BLOCK).map(|_| rng.byte()));
            bytes.extend([0u8; 2 * BLOCK]);

            let mut entries = AsyncTarEntries::new(futures_util::io::Cursor::new(bytes));
            // Read to exhaustion, or until it stops. Either is fine.
            for _ in 0..64 {
                match entries.next_entry().await {
                    Ok(Some(_)) => {}
                    Ok(None) | Err(_) => break,
                }
            }
        }

        // And the case that actually crashed: a header claiming more than the
        // machine can allocate.
        let mut block = corrupt_header(&mut rng);
        block[124..136].copy_from_slice(b"77777777777\0");
        let sum = checksum(&block);
        block[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        let mut entries = AsyncTarEntries::new(futures_util::io::Cursor::new(block.to_vec()));
        let _ = entries.next_entry().await;
    });
}
