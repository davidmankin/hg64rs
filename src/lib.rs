#![allow(dead_code, unused_variables)]

/*
 * hg64rs - 64-bit histograms
 *
 * Ported from Tony Finch's C language hg64.
 *
 *
 * C code Written by Tony Finch <dot@dotat.at> <fanf@isc.org>
 * You may do anything with this. It has no warranty.
 * <https://creativecommons.org/publicdomain/zero/1.0/>
 * SPDX-License-Identifier: CC0-1.0
 */

use std::mem;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

const KEYBITS: u16 = 12;
const EXPBITS: u16 = 6; // log2(VALUEBITS)
const MANBITS: u16 = KEYBITS - EXPBITS;
const MANSIZE: u16 = 1 << MANBITS;
const KEYSIZE: u16 = 1 << KEYBITS;
const PACKSIZE: u16 = 64;
const PACKS: u16 = KEYSIZE / PACKSIZE;

/*
 * We waste a little extra space in the PACKS array that could be saved
 * by omitting exponents that aren't needed by denormal numbers, but the
 * arithmetic gets awkward for smaller KEYBITS. However we need the
 * exact number of keys for accurate bounds checks.
 */
const DENORMALS: u16 = MANBITS - 1;
const EXPONENTS: u16 = KEYSIZE / MANSIZE - DENORMALS;
const KEYS: u16 = EXPONENTS * MANSIZE;

#[derive(Debug, Default, Clone)]
struct Pack {
    count: u64,
    bmp: u64, // Bitmap of which buckets exist (?)
    bucket: Vec<u64>,
}

#[derive(Debug)]
struct HG64 {
    packs: [Pack; PACKS as usize],
}

impl Default for HG64 {
    fn default() -> HG64 {
        HG64 {
            packs: array_init::array_init(|_| Pack::default()),
        }
    }
}

//fn is_divisible_by(lhs: u32, rhs: u32) -> bool {

fn interpolate(span: u64, mul: u64, div: u64) -> u64 {
    let frac = if div == 0 {
        1.0
    } else {
        (mul as f64) / (div as f64)
    };
    return (span as f64 * frac) as u64;
}

fn get_maxval(key: u16) -> u64 {
    // Source comment:
    // don't shift by 64; reduce shift by 2 and pre-shift UINT64_MAX
    let shift = PACKSIZE - key / MANSIZE - 1;
    let range = u64::MAX / 4 >> shift;
    get_minval(key) + range
}

fn get_minval(key: u16) -> u64 {
    let exponent = key / MANSIZE - 1;
    let mantissa = key % MANSIZE + MANSIZE;
    if key < MANSIZE {
        key as u64
    } else {
        (mantissa << exponent) as u64
    }
}

fn get_key(value: u64) -> u16 {
    // Original comment:
    // hot path
    if value < (MANSIZE as u64) {
        value as u16
    } else {
        // Built-in Function: int __builtin_clz (unsigned int x)
        // Returns the number of leading 0-bits in x, starting at the most significant bit position. If x is 0, the result is undefined.
        let clz = value.leading_zeros() as u16;
        let exponent = PACKSIZE - MANBITS - clz;
        let mantissa = (value >> (exponent - 1)) as u16;
        exponent * MANSIZE + mantissa % MANSIZE
    }
}

/*
 * Allocate a new histogram
 */
fn create_hg64() -> Box<HG64> {
    return Box::new(HG64::default());
}

impl HG64 {
    /*
     * Calculate the total count of all the buckets in the histogram
     */
    pub fn population(&self) -> u64 {
        let mut pop = 0u64;
        for pack in self.packs.iter() {
            pop += pack.count;
        }
        return pop;
    }

    /*
     * Calculate the number of buckets
     */
    pub fn buckets(&self) -> usize {
        let mut buckets: usize = 0;
        for pack in self.packs.iter() {
            // buckets += pack.bmp.count_ones() as usize;
            buckets += pack.bucket.len();
        }
        buckets
    }
    /*
     * Calculate the memory used in bytes
     */
    pub fn size(&self) -> usize {
        //TODO  this isn't right now that we use vec
        mem::size_of::<HG64>() + mem::size_of::<u64>() * self.buckets()
    }
    /*
     * Get the compile-time KEYBITS setting
     */
    pub fn keybits(&self) -> u16 {
        KEYBITS
    }

    /*
     * Add 1 to the value's bucket
     */
    pub fn inc(&mut self, value: u64) -> () {
        self.add(value, 1)
    }

    /*
     * Add an arbitrary count to the value's bucket
     */
    pub fn add(&mut self, value: u64, count: u64) -> () {
        if count > 0 {
            self.bump_count(get_key(value), count);
        }
    }

    /* TODO: THIS IS NOT IDIOMATIC RUST
     * Get information about a bucket. This can be used as an iterator, by
     * initializing `key` to zero and incrementing by one until `hg64_get()`
     * returns `false`.
     *
     * If `pmin` is non-NULL it is set to the bucket's minimum inclusive value.
     *
     * If `pmax` is non-NULL it is set to the bucket's maximum exclusive value.
     *
     * If `pcount` is non-NULL it is set to the bucket's counter, which
     * can be zero. (Empty buckets are included in the iterator.)
     */
    // source method: "hg64_get"
    pub fn get(&self, key: u64, pmin: &u64, pmax: &u64, pcount: &u64) -> bool {
        false
    }

    /*
     * Increase the counts in `self` by the counts recorded in `source`
     */
    // source method "hg64_merge"
    pub fn merge_from(&mut self, source: &HG64) -> () {
        for key in 0..KEYS {
            let count = source.get_key_count(key);
            self.bump_count(key, count)
        }
    }

    /*
     * Get the approximate value at a given rank in the recorded data.
     * The rank must be less than the histogram's population.
     */
    // source method: "hg64_value_at_rank"
    pub fn value_at_rank(&self, rank: u64) -> u64 {
        let mut rank: u64 = rank;
        let mut key: u16 = 0;
        while key < KEYS {
            let count: u64 = self.get_pack_count(key);
            if rank < count {
                break;
            }
            rank -= count;
            key += PACKSIZE;
        }
        if key == KEYS {
            return u64::MAX;
        }

        let stop = key + PACKSIZE;
        while key < stop {
            let count = self.get_key_count(key);
            if rank < count {
                break;
            }
            rank -= count;
            key += 1;
        }
        if key == KEYS {
            return u64::MAX;
        }

        let min: u64 = get_minval(key);
        let max: u64 = get_maxval(key);
        let count = self.get_key_count(key);
        return min + interpolate(max - min, rank, count);
    }

    /*
     * Get the approximate value at a given quantile in the recorded data.
     * The quantile must be >= 0.0 and < 1.0
     *
     * Quantiles are percentiles divided by 100. The median is quantile 1/2.
     */
    // uint64_t hg64_value_at_quantile(hg64 *hg, double quantile);
    pub fn value_at_quantile(&self, quanntile: f64) -> u64 {
        let pop: f64 = self.population() as f64;
        let q = if quanntile < 0.0 {
            0.0
        } else {
            if quanntile > 1.0 {
                1.0
            } else {
                quanntile
            }
        };
        let rank = q * pop;
        self.value_at_rank(rank as u64)
    }

    /*
     * Get the approximate rank of a value in the recorded data.
     * You can query the rank of any value.
     */
    // uint64_t hg64_rank_of_value(hg64 *hg, uint64_t value);
    pub fn rank_of_value(&self, value: u64) -> u64 {
        let key = get_key(value);
        let k0 = key - key % PACKSIZE;
        let mut rank: u64 = 0;

        for k in (0..k0).step_by(PACKSIZE as usize) {
            rank += self.get_pack_count(k);
        }
        for k in k0..key {
            rank += self.get_key_count(k);
        }
        let count = self.get_key_count(key);
        let min = get_minval(key);
        let max = get_maxval(key);
        rank + interpolate(count, value - min, max - min)
    }

    /*
     * Get the approximate quantile of a value in the recorded data.
     */
    // double hg64_quantile_of_value(hg64 *hg, uint64_t value);
    pub fn quantile_of_value(&self, value: u64) -> f64 {
        let rank = self.rank_of_value(value);
        return rank as f64 / self.population() as f64;
    }

    /* // Actual rust interface:
     * Get summary statistics about the histogram. Returns a tuple with
     * (mean of recorded data, variannce of recorded data)
     */
    /* // Originnal comment:
     * Get summary statistics about the histogram.
     *
     * If `pmean` is non-NULL it is set to the mean of the recorded data.
     *
     * If `pvar` is non-NULL it is set to the variance of the recorded
     * data. The standard deviation is the square root of the variance.
     */
    // void hg64_mean_variance(hg64 *hg, double *pmean, double *pvar);
    pub fn mean_variance(&self) -> (f64, f64) {
        let mut pop = 0.0;
        let mut mean = 0.0;
        let mut sigma = 0.0;

        for key in 0..KEYS {
            let min = get_minval(key) as f64 / 2.0;
            let max = get_maxval(key) as f64 / 2.0;
            let count = self.get_key_count(key);
            let delta = min + max - mean;
            if count != 0 {
                let count = count as f64;
                pop += count;
                mean += count * delta / pop;
                sigma += count * delta * (min + max - mean);
            }
        }
        return (mean, sigma / pop);
    }

    // ================ internal ==============

    fn validate(&self) -> () {
        let min = 0_u64;
        let max = 1_u64 << 16;
        let step = 1_u64;
        for value in (0..max).step_by(step as usize) {
            self.validate_value(value);
        }
        let min = 1_u64 << 30;
        let max = 1_u64 << 40;
        let step = 1_u64 << 20;
        for value in (0..max).step_by(step as usize) {
            self.validate_value(value);
        }
        let max = u64::MAX;
        let min = max >> 8;
        let step = max >> 10;
        for value in ((min + 1)..max).step_by(step as usize) {
            self.validate_value(value);
        }
        for key in 1..KEYS {
            assert!(get_maxval(key - 1) < get_minval(key))
        }
        for p in 0..PACKS as usize {
            let mut count = 0_u64;
            let pack = &self.packs[p];
            let pop = pack.bmp.count_ones() as usize;
            let len = pack.bucket.len();
            assert_eq!(pop, len);
            for pos in 0..pop {
                assert_ne!(pack.bucket[pos], 0);
                count += pack.bucket[pos];
            }
            assert_eq!(count == 0, pack.bucket.capacity() == 0);
            assert_eq!(count == 0, pack.bmp == 0);
            assert_eq!(count, pack.count);
        }
    }

    fn validate_value(&self, value: u64) {
        let key = get_key(value);
        let min = get_minval(key);
        let max = get_maxval(key);
        assert!(key < KEYS);
        assert!(key / PACKSIZE < PACKS);
        assert!(value >= min);
        assert!(value <= max);
    }

    fn bump_count(&mut self, key: u16, count: u64) -> () {
        self.packs[(key / PACKSIZE) as usize].count += count;
        let (pack_index, pos) = self.find_bucket(key);
        self.packs[pack_index].bucket[pos] += count;
    }

    fn get_key_count(&self, key: u16) -> u64 {
        self.get_bucket_value(key)
    }

    fn get_pack_count(&self, key: u16) -> u64 {
        self.packs[(key / PACKSIZE) as usize].count
    }

    // Original name: "uint64_t *get_bucket"
    // Original comment:
    // Here we have fun indexing into a pack, and expanding if if necessary.
    //
    // My comments on the C version:
    // Each pack holds zero or more buckets.  Each bucket is a single u64
    // count of how many measurements have been seen in this bucket.
    // The pack.bmp is a bitmap that says which buckets are allocated.
    // e.g. in binary 0101000 would mean that the buckets array is two long
    // and the first item represents the 8's place in the pack, and the second
    // bucket represents the 32's place.
    // When we look for an existing bucket we can check if it already exists;
    // if so we return a pointer to that spot in the buckets array for mutation.
    // If it doesn't already exist: In "nullable" mode (i.e. non-mutable mode),
    // null is returned instead of the pointer.
    // In non-nullable mode, we allocate a new bucket by lengthening the
    // buckets array by one, moving the higher items over one using memmove
    // updating the bmp map,
    // and returning the pointer to the newly inserted zero.
    //
    // My comments on Rust version:
    // Instead of using popcount on the bmp to remember how long the bucket
    // array is, we'll just suck it up and keep an int for each pack.
    // This growable array that knows its own length is a Vec, so that's
    // what we'll use as the type for the list of buckets inside a pack.
    // We will still maintain the bitmap, and to return a mutable pointer
    // to the bucket we'll return a boxed single element slice from the vec.
    fn get_bucket_value(&self, key: u16) -> u64 {
        let pack = &self.packs[(key / PACKSIZE) as usize];
        let bit = 1u64 << (key % PACKSIZE);
        let mask = bit - 1;
        let bmp = pack.bmp;
        let pos: usize = (bmp & mask).count_ones() as usize;
        if bmp & bit != 0 {
            return pack.bucket[pos];
        } else {
            return 0;
        }
    }
    // Original name: get_bucket
    // Returns the pack index & bucket offset within theh pack
    // for the given key. Only valid until a future call to
    // find_bucket.
    fn find_bucket(&mut self, key: u16) -> (usize, usize) {
        // Original comment: hot path
        let pack_index = (key / PACKSIZE) as usize;
        let pack = &mut self.packs[pack_index];
        let bit = 1u64 << (key % PACKSIZE);
        let mask = bit - 1;
        let bmp = pack.bmp;
        let pos: usize = (bmp & mask).count_ones() as usize;
        if bmp & bit != 0 {
            // How to return a mutable pointer to an item in the array?
            // return pack.bucket[pos];
            return (pack_index, pos);
        }

        // Original comment: cold path

        // Originally there was an optional "nullable" parameter that would
        // just return null instead of mutating the packs.  But for now we
        // remove it for simplicity in converting to Rustl
        // if nullable {
        //     return null;
        // }
        // let pop = bmp.count_ones() as usize;
        let len = pack.bucket.len();
        // assert_eq!(pop, len);
        // let need = len + 1;
        // let move_ = len - pos;
        // Original C code
        // 	uint64_t *ptr = realloc(pack->bucket, need * sizeof(uint64_t));
        // memmove(ptr + pos + 1, ptr + pos, move * sizeof(uint64_t));
        // pack->bucket = ptr;
        // pack->bucket[pos] = 0
        pack.bucket.insert(pos, 0);
        pack.bmp |= bit;
        // return &pack->bucket[pos]
        return (pack_index, pos);
    }
}