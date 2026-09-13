//! Collating samples into batches.
//!
//! Batching a stream of samples means turning `n` maps of field to value into
//! one map of field to a stacked value. Fields that are numeric stack into a
//! single [`Tensor`] with an extra leading dimension; anything else becomes a
//! [`Value::List`], which is the same rule the Python `default_collation_fn`
//! follows.
//!
//! ```
//! use webdataset::batch::collate_samples;
//! use webdataset_core::{Sample, Value};
//!
//! let samples: Vec<Sample> = (0..4)
//!     .map(|i| {
//!         let mut s = Sample::with_key(format!("k{i}"));
//!         s.insert("cls", Value::Int(i));
//!         s
//!     })
//!     .collect();
//!
//! let batch = collate_samples(samples)?;
//! assert_eq!(batch.get("cls").unwrap().as_tensor().unwrap().shape(), &[4]);
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use bytes::{BufMut, BytesMut};
use webdataset_core::error::{Error, Result};
use webdataset_core::sample::Sample;
use webdataset_core::tensor::{DType, Tensor};
use webdataset_core::value::Value;

/// Stack a batch of samples field by field.
///
/// Every sample must have the same fields; a mismatch is an error rather than a
/// silently dropped column.
pub fn collate_samples(samples: Vec<Sample>) -> Result<Sample> {
    let Some(first) = samples.first() else {
        return Err(Error::value("cannot collate an empty batch"));
    };
    let names: Vec<String> = first.keys().map(str::to_string).collect();

    for (i, sample) in samples.iter().enumerate().skip(1) {
        if sample.len() != names.len() || names.iter().any(|n| !sample.contains_key(n)) {
            return Err(Error::value(format!(
                "sample {i} has fields {:?} but the batch expects {names:?}",
                sample.keys().collect::<Vec<_>>()
            )));
        }
    }

    let mut out = Sample::new();
    for name in &names {
        let column: Vec<&Value> = samples.iter().map(|s| s.get(name).expect("checked above")).collect();
        out.insert(name.clone(), collate_values(&column)?);
    }
    Ok(out)
}

/// Stack a batch of tuples element by element.
pub fn collate_tuples(rows: Vec<Vec<Value>>) -> Result<Vec<Value>> {
    let Some(first) = rows.first() else {
        return Err(Error::value("cannot collate an empty batch"));
    };
    let width = first.len();
    if let Some(bad) = rows.iter().position(|row| row.len() != width) {
        return Err(Error::value(format!("row {bad} has {} values but the batch expects {width}", rows[bad].len())));
    }

    (0..width)
        .map(|i| {
            let column: Vec<&Value> = rows.iter().map(|row| &row[i]).collect();
            collate_values(&column)
        })
        .collect()
}

/// Stack one column of a batch.
///
/// Tensors of identical shape and type stack into one tensor; integers and
/// floats become a one-dimensional tensor; everything else stays a list.
pub fn collate_values(column: &[&Value]) -> Result<Value> {
    let Some(first) = column.first() else {
        return Ok(Value::List(Vec::new()));
    };
    match first {
        Value::Tensor(_) => stack_tensors(column),
        Value::Int(_) | Value::Bool(_) if column.iter().all(|v| v.as_i64().is_some()) => {
            let values: Vec<i64> = column.iter().filter_map(|v| v.as_i64()).collect();
            let mut data = BytesMut::with_capacity(values.len() * 8);
            for v in &values {
                data.put_slice(&v.to_ne_bytes());
            }
            Ok(Value::Tensor(Tensor::new(DType::I64, vec![values.len()], data.freeze())?))
        }
        Value::Float(_) if column.iter().all(|v| v.as_f64().is_some()) => {
            let values: Vec<f64> = column.iter().filter_map(|v| v.as_f64()).collect();
            let mut data = BytesMut::with_capacity(values.len() * 8);
            for v in &values {
                data.put_slice(&v.to_ne_bytes());
            }
            Ok(Value::Tensor(Tensor::new(DType::F64, vec![values.len()], data.freeze())?))
        }
        _ => Ok(Value::List(column.iter().map(|v| (*v).clone()).collect())),
    }
}

/// Stack tensors that share a shape and element type.
fn stack_tensors(column: &[&Value]) -> Result<Value> {
    let tensors: Vec<&Tensor> = column.iter().filter_map(|v| v.as_tensor()).collect();
    if tensors.len() != column.len() {
        // A mixed column cannot be stacked; keep it as a list.
        return Ok(Value::List(column.iter().map(|v| (*v).clone()).collect()));
    }
    let first = tensors[0];
    let uniform = tensors.iter().all(|t| t.shape() == first.shape() && t.dtype() == first.dtype());
    if !uniform {
        return Err(Error::value(format!(
            "cannot stack tensors of differing shapes: {:?}",
            tensors.iter().map(|t| t.shape()).collect::<Vec<_>>()
        )));
    }

    let mut data = BytesMut::with_capacity(first.data().len() * tensors.len());
    for tensor in &tensors {
        data.put_slice(tensor.data());
    }
    let mut shape = vec![tensors.len()];
    shape.extend_from_slice(first.shape());
    Ok(Value::Tensor(Tensor::new(first.dtype(), shape, data.freeze())?))
}

/// Split a collated batch back into individual samples.
///
/// This is the inverse of [`collate_samples`] for columns that were stacked;
/// list columns are split element by element.
pub fn uncollate_sample(batch: &Sample) -> Result<Vec<Sample>> {
    let mut size = None;
    for (name, value) in batch {
        let n = match value {
            Value::Tensor(t) if !t.shape().is_empty() => t.shape()[0],
            Value::List(items) => items.len(),
            other => {
                return Err(Error::value(format!("field {name} is {}, which is not batched", other.type_name())));
            }
        };
        match size {
            None => size = Some(n),
            Some(m) if m != n => {
                return Err(Error::value(format!("field {name} has {n} rows but the batch has {m}")));
            }
            Some(_) => {}
        }
    }
    let size = size.unwrap_or(0);

    let mut out = vec![Sample::new(); size];
    for (name, value) in batch {
        for (i, sample) in out.iter_mut().enumerate() {
            sample.insert(name.clone(), row(value, i)?);
        }
    }
    Ok(out)
}

/// Extract row `index` from a stacked value.
fn row(value: &Value, index: usize) -> Result<Value> {
    match value {
        Value::List(items) => Ok(items[index].clone()),
        Value::Tensor(t) => {
            let inner: Vec<usize> = t.shape()[1..].to_vec();
            let stride = inner.iter().product::<usize>() * t.dtype().size();
            let start = index * stride;
            let slice = t.data().slice(start..start + stride);
            // A stacked scalar column becomes a scalar again.
            if inner.is_empty() {
                let scalar = Tensor::new(t.dtype(), vec![], slice)?;
                return Ok(match t.dtype().is_float() {
                    true => Value::Float(scalar.get(0)),
                    false => Value::Int(scalar.get(0) as i64),
                });
            }
            Ok(Value::Tensor(Tensor::new(t.dtype(), inner, slice)?))
        }
        other => Err(Error::value(format!("cannot unbatch {}", other.type_name()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch_of(n: i64) -> Vec<Sample> {
        (0..n)
            .map(|i| {
                let mut s = Sample::with_key(format!("k{i}"));
                s.insert("cls", Value::Int(i));
                s.insert("vec", Value::Tensor(Tensor::from_f32(&[i as f32, 0.5])));
                s.insert("txt", Value::Text(format!("text {i}")));
                s
            })
            .collect()
    }

    #[test]
    fn stacks_integers_into_a_tensor() {
        let batch = collate_samples(batch_of(4)).unwrap();
        let cls = batch.get("cls").unwrap().as_tensor().unwrap();
        assert_eq!(cls.shape(), &[4]);
        assert_eq!(cls.dtype(), DType::I64);
        assert_eq!(cls.to_f64_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn stacks_tensors_with_a_leading_dimension() {
        let batch = collate_samples(batch_of(3)).unwrap();
        let vectors = batch.get("vec").unwrap().as_tensor().unwrap();
        assert_eq!(vectors.shape(), &[3, 2]);
        assert_eq!(vectors.to_f64_vec(), vec![0.0, 0.5, 1.0, 0.5, 2.0, 0.5]);
    }

    #[test]
    fn keeps_text_columns_as_lists() {
        let batch = collate_samples(batch_of(2)).unwrap();
        let texts = batch.get("txt").unwrap().as_list().unwrap();
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[1], Value::Text("text 1".into()));
        assert_eq!(batch.get("__key__").unwrap().as_list().unwrap().len(), 2);
    }

    #[test]
    fn refuses_ragged_batches() {
        let mut samples = batch_of(2);
        samples[1].remove("txt");
        let err = collate_samples(samples).unwrap_err();
        assert!(err.to_string().contains("expects"), "{err}");

        let mut samples = batch_of(2);
        samples[1].insert("vec", Value::Tensor(Tensor::from_f32(&[1.0, 2.0, 3.0])));
        assert!(collate_samples(samples).is_err(), "tensors of different shapes cannot stack");
    }

    #[test]
    fn refuses_empty_batches() {
        assert!(collate_samples(Vec::new()).is_err());
        assert!(collate_tuples(Vec::new()).is_err());
    }

    #[test]
    fn collates_tuples_columnwise() {
        let rows = vec![vec![Value::Int(1), Value::Text("a".into())], vec![Value::Int(2), Value::Text("b".into())]];
        let batch = collate_tuples(rows).unwrap();
        assert_eq!(batch[0].as_tensor().unwrap().shape(), &[2]);
        assert_eq!(batch[1].as_list().unwrap().len(), 2);
    }

    #[test]
    fn round_trips_through_uncollate() {
        let samples = batch_of(3);
        let batch = collate_samples(samples.clone()).unwrap();
        let back = uncollate_sample(&batch).unwrap();

        assert_eq!(back.len(), 3);
        assert_eq!(back[1].get("cls").unwrap().as_i64(), Some(1));
        assert_eq!(back[1].get("txt").unwrap(), &Value::Text("text 1".into()));
        assert_eq!(back[2].get("vec").unwrap().as_tensor().unwrap(), &Tensor::from_f32(&[2.0, 0.5]));
    }
}
