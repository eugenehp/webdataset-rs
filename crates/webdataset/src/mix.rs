//! Combining several datasets into one stream.
//!
//! Training on a mixture of sources — a large generic corpus plus a small
//! domain-specific one, say — means interleaving several pipelines.
//! [`RoundRobin`] takes from each in turn; [`RandomMix`] draws from them at
//! given probabilities, which is how a small source is up-weighted without
//! physically duplicating its shards.
//!
//! ```
//! use webdataset::mix::RandomMix;
//! use webdataset::pipeline::{DataPipeline, Samples};
//! use webdataset_core::Sample;
//!
//! let a = DataPipeline::new().with(Samples::new((0..100).map(|i| Sample::with_key(format!("a{i}")))));
//! let b = DataPipeline::new().with(Samples::new((0..100).map(|i| Sample::with_key(format!("b{i}")))));
//!
//! let mixed = DataPipeline::new().with(RandomMix::new(vec![a, b]).with_weights(&[0.9, 0.1])?);
//! assert!(mixed.iter().count() > 0);
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use std::sync::Mutex;

use rand::prelude::*;
use rand::rngs::StdRng;
use webdataset_core::error::{Error, Result};

use crate::pipeline::{DataPipeline, SampleStream, Stage};

/// Takes one sample from each dataset in turn.
#[derive(Debug)]
pub struct RoundRobin {
    datasets: Vec<DataPipeline>,
    longest: bool,
}

impl RoundRobin {
    /// Interleave these datasets, stopping when the first one runs out.
    pub fn new(datasets: Vec<DataPipeline>) -> RoundRobin {
        RoundRobin { datasets, longest: false }
    }

    /// Keep going until every dataset has run out.
    pub fn longest(mut self) -> RoundRobin {
        self.longest = true;
        self
    }
}

impl Stage for RoundRobin {
    fn apply(&self, _input: SampleStream) -> SampleStream {
        let mut sources: Vec<SampleStream> = self.datasets.iter().map(DataPipeline::iter).collect();
        let longest = self.longest;
        let mut next = 0usize;

        Box::new(std::iter::from_fn(move || {
            while !sources.is_empty() {
                next %= sources.len();
                match sources[next].next() {
                    Some(item) => {
                        next += 1;
                        return Some(item);
                    }
                    None if longest => {
                        // Drop the exhausted source; the others continue.
                        drop(sources.remove(next));
                    }
                    None => return None,
                }
            }
            None
        }))
    }
}

/// Draws from several datasets at given probabilities.
#[derive(Debug)]
pub struct RandomMix {
    datasets: Vec<DataPipeline>,
    weights: Vec<f64>,
    longest: bool,
    seed: Option<u64>,
}

impl RandomMix {
    /// Mix these datasets with equal weight.
    pub fn new(datasets: Vec<DataPipeline>) -> RandomMix {
        let weights = vec![1.0; datasets.len()];
        RandomMix { datasets, weights, longest: false, seed: None }
    }

    /// Weight each dataset; weights need not sum to one.
    pub fn with_weights(mut self, weights: &[f64]) -> Result<RandomMix> {
        if weights.len() != self.datasets.len() {
            return Err(Error::value(format!("got {} weights for {} datasets", weights.len(), self.datasets.len())));
        }
        if weights.iter().any(|w| *w < 0.0) || weights.iter().sum::<f64>() <= 0.0 {
            return Err(Error::value("weights must be non-negative and not all zero"));
        }
        self.weights = weights.to_vec();
        Ok(self)
    }

    /// Keep going until every dataset has run out, rather than stopping at the first.
    pub fn longest(mut self) -> RandomMix {
        self.longest = true;
        self
    }

    /// Make the draw reproducible.
    pub fn with_seed(mut self, seed: u64) -> RandomMix {
        self.seed = Some(seed);
        self
    }
}

impl Stage for RandomMix {
    fn apply(&self, _input: SampleStream) -> SampleStream {
        let mut sources: Vec<SampleStream> = self.datasets.iter().map(DataPipeline::iter).collect();
        let mut weights = self.weights.clone();
        let longest = self.longest;
        let rng = Mutex::new(match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rand::rng().random()),
        });

        Box::new(std::iter::from_fn(move || {
            while !sources.is_empty() {
                let index = {
                    let mut rng = rng.lock().expect("rng lock");
                    weighted_choice(&weights, rng.random::<f64>())
                };
                match sources[index].next() {
                    Some(item) => return Some(item),
                    None if longest => {
                        drop(sources.remove(index));
                        weights.remove(index);
                        if weights.iter().sum::<f64>() <= 0.0 {
                            return None;
                        }
                    }
                    None => return None,
                }
            }
            None
        }))
    }
}

/// Pick an index in proportion to `weights`, given a uniform draw in `0..1`.
fn weighted_choice(weights: &[f64], uniform: f64) -> usize {
    let total: f64 = weights.iter().sum();
    let mut cumulative = 0.0;
    for (i, weight) in weights.iter().enumerate() {
        cumulative += weight / total;
        if uniform < cumulative {
            return i;
        }
    }
    weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Samples;
    use webdataset_core::Sample;

    fn dataset(prefix: &str, n: usize) -> DataPipeline {
        let prefix = prefix.to_string();
        DataPipeline::new().with(Samples::new((0..n).map(move |i| Sample::with_key(format!("{prefix}{i}")))))
    }

    fn keys(pipeline: &DataPipeline) -> Vec<String> {
        pipeline.iter().map(|s| s.unwrap().key().unwrap().to_string()).collect()
    }

    #[test]
    fn round_robin_interleaves() {
        let mixed = DataPipeline::new().with(RoundRobin::new(vec![dataset("a", 3), dataset("b", 3)]));
        assert_eq!(keys(&mixed), ["a0", "b0", "a1", "b1", "a2", "b2"]);
    }

    #[test]
    fn round_robin_stops_at_the_shortest_by_default() {
        let mixed = DataPipeline::new().with(RoundRobin::new(vec![dataset("a", 1), dataset("b", 5)]));
        assert_eq!(keys(&mixed), ["a0", "b0"]);
    }

    #[test]
    fn round_robin_can_drain_the_longest() {
        let mixed = DataPipeline::new().with(RoundRobin::new(vec![dataset("a", 1), dataset("b", 3)]).longest());
        assert_eq!(keys(&mixed), ["a0", "b0", "b1", "b2"]);
    }

    #[test]
    fn random_mix_respects_weights() {
        let mixed = DataPipeline::new().with(
            RandomMix::new(vec![dataset("a", 10000), dataset("b", 10000)])
                .with_weights(&[0.9, 0.1])
                .unwrap()
                .with_seed(1),
        );

        let drawn = keys(&mixed);
        let from_a = drawn.iter().filter(|k| k.starts_with('a')).count();
        let ratio = from_a as f64 / drawn.len() as f64;
        assert!((0.85..0.95).contains(&ratio), "drew {ratio:.2} from a, expected about 0.9");
    }

    #[test]
    fn random_mix_is_reproducible() {
        let build =
            || DataPipeline::new().with(RandomMix::new(vec![dataset("a", 100), dataset("b", 100)]).with_seed(7));
        assert_eq!(keys(&build()), keys(&build()));
    }

    #[test]
    fn random_mix_rejects_bad_weights() {
        assert!(RandomMix::new(vec![dataset("a", 1)]).with_weights(&[1.0, 1.0]).is_err());
        assert!(RandomMix::new(vec![dataset("a", 1)]).with_weights(&[0.0]).is_err());
        assert!(RandomMix::new(vec![dataset("a", 1)]).with_weights(&[-1.0]).is_err());
    }

    #[test]
    fn random_mix_can_drain_the_longest() {
        let mixed =
            DataPipeline::new().with(RandomMix::new(vec![dataset("a", 2), dataset("b", 50)]).longest().with_seed(3));
        assert_eq!(keys(&mixed).len(), 52);
    }

    #[test]
    fn weighted_choice_follows_the_cumulative_distribution() {
        assert_eq!(weighted_choice(&[0.5, 0.5], 0.1), 0);
        assert_eq!(weighted_choice(&[0.5, 0.5], 0.9), 1);
        assert_eq!(weighted_choice(&[1.0, 0.0], 0.99), 0);
        assert_eq!(weighted_choice(&[0.0, 1.0], 0.0), 1);
    }
}
