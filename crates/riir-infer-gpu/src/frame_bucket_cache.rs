//! Issue 1238 — the frame-size-keyed buffer cache, BUCKETED by row count.
//!
//! Every per-frame scratch the prefill GEMM lanes keep (the zero-copy
//! GEMM's dedicated output, the f16-B twin's cast input, the i1226 F2
//! producer-shadow pool) used to be keyed by its EXACT byte length and
//! never evicted. A serving process sees every distinct final-frame (tail)
//! length, so each new tail added one buffer per key for the process
//! lifetime — the Issue 1238 T1 census on Bonsai-2-27B: 1.72 GiB worst case
//! at chunk 128, 27.4 GiB at chunk 512, against a 0.03 / 0.13 GiB
//! fixed-width floor.
//!
//! The bound: the key is `(row_bytes, rows.next_power_of_two())` and the
//! buffer is allocated at the BUCKET size, so a tail reuses the buffer of
//! its power-of-two class. Per row width that is at most
//! `log2(max_rows) + 1` live buffers summing to `< 2 × bucket(max_rows)`
//! rows — resident bytes are bounded by `4 × the full-frame floor` whatever
//! the prompt-length mix.
//!
//! Byte identity is by construction: every consumer passes the EXACT
//! extent (`from_raw_parts(.., total)`, the blit's `output_bytes`, the
//! gemm's `p`/`m`/`n` uniforms), so the slack tail of a bucket is never read
//! back. Keeping `row_bytes` in the key keeps today's aliasing classes:
//! inside one frame (one `rows`), two different row widths never share a
//! buffer — the same per-frame sharing the exact-byte key gave, minus its
//! accidental cross-width collisions (`p1·w1 == p2·w2`).
//!
//! Pure std (no device types) so the bound is unit-tested on every host.

use std::collections::HashMap;
use std::sync::Mutex;

/// The row-count class a frame of `rows` rows is served from (`rows == 0`
/// maps to class 1 — a zero-row frame never allocates a zero-size buffer).
#[inline]
pub fn frame_rows_bucket(rows: usize) -> usize {
    rows.max(1).next_power_of_two()
}

/// A `(row_bytes, bucketed rows)`-keyed cache of device buffers (any
/// cloneable handle type — `metal::Buffer`, a CubeCL `Handle`).
pub struct FrameBucketCache<B> {
    map: Mutex<HashMap<(u64, usize), (u64, B)>>,
}

impl<B: Clone> Default for FrameBucketCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Clone> FrameBucketCache<B> {
    pub fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }

    /// The buffer serving a frame of `rows` rows of `row_bytes` bytes each,
    /// allocating `alloc(bucket_bytes)` on the class's first use. The
    /// returned buffer is AT LEAST `row_bytes × rows` bytes; callers bind it
    /// with their exact extent.
    pub fn get_or_alloc(&self, row_bytes: u64, rows: usize, alloc: impl FnOnce(u64) -> B) -> B {
        let bucket = frame_rows_bucket(rows);
        let mut map = self.map.lock().expect("frame bucket cache");
        map.entry((row_bytes, bucket))
            .or_insert_with(|| {
                let bytes = row_bytes * bucket as u64;
                (bytes, alloc(bytes))
            })
            .1
            .clone()
    }

    /// The buffer a frame of `rows` rows of `row_bytes` would be served
    /// from, if its class has been allocated (the tap's read-back lookup).
    pub fn get(&self, row_bytes: u64, rows: usize) -> Option<B> {
        let map = self.map.lock().expect("frame bucket cache");
        map.get(&(row_bytes, frame_rows_bucket(rows)))
            .map(|(_, b)| b.clone())
    }

    /// `(live buffers, resident bytes)` — the Issue 1238 census reading.
    pub fn census(&self) -> (usize, u64) {
        let map = self.map.lock().expect("frame bucket cache");
        (map.len(), map.values().map(|(bytes, _)| bytes).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The serving tail sweep (Issue 1238 T1's mix): tails 1..=chunk at the
    /// three Bonsai-2-27B rotate widths. Key count is logarithmic and
    /// resident bytes stay under 4× the full-frame floor.
    #[test]
    fn tail_sweep_is_bounded() {
        for chunk in [128usize, 384, 512, 4096] {
            let cache: FrameBucketCache<u64> = FrameBucketCache::new();
            let widths_bytes = [5120u64 * 2, 6144 * 2, 17408 * 2];
            for p in 1..=chunk {
                for &rb in &widths_bytes {
                    let got = cache.get_or_alloc(rb, p, |bytes| bytes);
                    assert!(got >= rb * p as u64, "bucket smaller than frame");
                }
            }
            let (keys, bytes) = cache.census();
            let classes = frame_rows_bucket(chunk).trailing_zeros() as usize + 1;
            assert_eq!(keys, classes * widths_bytes.len());
            let floor: u64 = widths_bytes.iter().map(|rb| rb * chunk as u64).sum();
            assert!(bytes < 4 * floor, "chunk {chunk}: {bytes} >= 4 x {floor}");
        }
    }

    /// Steady state at a fixed width allocates exactly once (G4).
    #[test]
    fn fixed_width_allocates_once() {
        let cache: FrameBucketCache<u64> = FrameBucketCache::new();
        let mut allocs = 0;
        for _ in 0..100 {
            cache.get_or_alloc(10240, 128, |b| {
                allocs += 1;
                b
            });
        }
        assert_eq!(allocs, 1);
        assert_eq!(cache.census(), (1, 10240 * 128));
    }

    /// Row width stays in the key: same rows, different widths never alias,
    /// and a tail reuses its class's buffer.
    #[test]
    fn widths_never_alias_and_tails_share_class() {
        let cache: FrameBucketCache<(u64, u64)> = FrameBucketCache::new();
        let a = cache.get_or_alloc(10240, 96, |b| (10240, b));
        let b = cache.get_or_alloc(12288, 96, |b| (12288, b));
        assert_ne!(a.0, b.0);
        let tail = cache.get_or_alloc(10240, 65, |_| unreachable!("class 128 exists"));
        assert_eq!(tail, a);
        assert_eq!(cache.get(10240, 100), Some(a));
        assert_eq!(cache.get(10240, 64), None);
        assert_eq!(frame_rows_bucket(0), 1);
    }
}
