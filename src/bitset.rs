#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BitMask {
    len: usize,
    words: Vec<u64>,
}

impl BitMask {
    pub fn new_zeros(len: usize) -> Self {
        let n_words = len.div_ceil(64);
        Self {
            len,
            words: vec![0u64; n_words],
        }
    }

    pub fn from_bool_iter<I>(len: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = bool>,
    {
        let mut mask = Self::new_zeros(len);
        for (idx, flag) in iter.into_iter().enumerate() {
            if flag {
                mask.set(idx, true);
            }
        }
        mask
    }

    pub fn set(&mut self, idx: usize, value: bool) {
        let word = idx / 64;
        let bit = idx % 64;
        if value {
            self.words[word] |= 1u64 << bit;
        } else {
            self.words[word] &= !(1u64 << bit);
        }
    }

    pub fn get(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        ((self.words[word] >> bit) & 1u64) == 1u64
    }

    pub fn and(&self, other: &Self) -> Self {
        debug_assert_eq!(self.len, other.len);
        let words = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| a & b)
            .collect::<Vec<_>>();
        Self {
            len: self.len,
            words,
        }
    }

    pub fn and_count(&self, other: &Self) -> usize {
        debug_assert_eq!(self.len, other.len);
        self.words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| (a & b).count_ones() as usize)
            .sum()
    }

    pub fn not(&self) -> Self {
        let words = self.words.iter().map(|w| !w).collect::<Vec<_>>();
        let mut out = Self {
            len: self.len,
            words,
        };
        out.clear_excess_bits();
        out
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    #[cfg(test)]
    pub fn to_bools(&self) -> Vec<bool> {
        (0..self.len).map(|i| self.get(i)).collect()
    }

    pub fn indices(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.count_ones());
        for (word_idx, word) in self.words.iter().enumerate() {
            let mut w = *word;
            while w != 0 {
                let tz = w.trailing_zeros() as usize;
                let idx = word_idx * 64 + tz;
                if idx < self.len {
                    out.push(idx);
                }
                w &= w - 1;
            }
        }
        out
    }

    fn clear_excess_bits(&mut self) {
        let excess = self.words.len() * 64 - self.len;
        if excess > 0 {
            let keep = 64 - excess;
            let mask = if keep == 64 {
                u64::MAX
            } else {
                (1u64 << keep) - 1u64
            };
            if let Some(last) = self.words.last_mut() {
                *last &= mask;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BitMask;

    #[test]
    fn basic_bitmask_ops() {
        let a = BitMask::from_bool_iter(6, [true, false, true, false, true, false]);
        let b = BitMask::from_bool_iter(6, [true, true, false, false, true, false]);
        let c = a.and(&b);
        assert_eq!(c.to_bools(), vec![true, false, false, false, true, false]);
        assert_eq!(c.count_ones(), 2);
        assert_eq!(a.and_count(&b), 2);
        assert_eq!(
            a.not().to_bools(),
            vec![false, true, false, true, false, true]
        );
    }
}
