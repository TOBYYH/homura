//! [`StateDict`] and the results of [`Module::load_state_dict`](super::Module::load_state_dict).

use std::collections::HashMap;
use std::fmt;

use crate::Tensor;

/// The parameters and persistent buffers of a module by name, in the module's order, like the
/// `OrderedDict` that PyTorch's `state_dict()` returns.
///
/// [`Module::state_dict`](super::Module::state_dict) fills it; it can also be collected from
/// `(String, Tensor)` pairs, such as those [`load_safetensors`](crate::load_safetensors)
/// returns, and iterated by reference to save it with
/// [`save_safetensors`](crate::save_safetensors).
///
/// ```
/// use homura::nn::StateDict;
///
/// let mut sd = StateDict::new();
/// sd.insert("weight", homura::zeros([2, 3]));
/// sd.insert("bias", homura::zeros([2]));
/// assert_eq!(sd.keys().collect::<Vec<_>>(), ["weight", "bias"]);
/// let bytes = homura::save_safetensors_to_bytes(&sd).unwrap();
/// let loaded: StateDict = homura::load_safetensors_from_bytes(&bytes)
///     .unwrap()
///     .into_iter()
///     .collect();
/// assert_eq!(loaded["bias"].shape(), [2]);
/// ```
#[derive(Clone, Default)]
pub struct StateDict {
    entries: Vec<(String, Tensor)>,
    index: HashMap<String, usize>,
}

impl StateDict {
    /// An empty state dict.
    pub fn new() -> StateDict {
        StateDict::default()
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The tensor stored under `key`.
    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.index.get(key).map(|&i| &self.entries[i].1)
    }

    /// Whether there is an entry for `key`.
    pub fn contains_key(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// Stores `value` under `key` and returns the tensor stored there before, if any. A new key
    /// goes last; an existing one keeps its position.
    pub fn insert(&mut self, key: impl Into<String>, value: Tensor) -> Option<Tensor> {
        let key = key.into();
        match self.index.get(&key) {
            Some(&i) => Some(std::mem::replace(&mut self.entries[i].1, value)),
            None => {
                self.index.insert(key.clone(), self.entries.len());
                self.entries.push((key, value));
                None
            }
        }
    }

    /// Removes the entry for `key` and returns its tensor, keeping the order of the others.
    pub fn remove(&mut self, key: &str) -> Option<Tensor> {
        let i = self.index.remove(key)?;
        let (_, value) = self.entries.remove(i);
        for position in self.index.values_mut() {
            if *position > i {
                *position -= 1;
            }
        }
        Some(value)
    }

    /// The keys, in order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }

    /// The tensors, in order.
    pub fn values(&self) -> impl Iterator<Item = &Tensor> {
        self.entries.iter().map(|(_, v)| v)
    }

    /// The entries, in order.
    pub fn iter(&self) -> StateDictIter<'_> {
        StateDictIter(self.entries.iter())
    }
}

/// The tensor stored under a key.
///
/// # Panics
///
/// Panics if there is no entry for the key.
impl std::ops::Index<&str> for StateDict {
    type Output = Tensor;

    fn index(&self, key: &str) -> &Tensor {
        self.get(key)
            .unwrap_or_else(|| panic!("no entry {key:?} in the state dict"))
    }
}

impl<K: Into<String>> FromIterator<(K, Tensor)> for StateDict {
    fn from_iter<I: IntoIterator<Item = (K, Tensor)>>(iter: I) -> Self {
        let mut sd = StateDict::new();
        sd.extend(iter);
        sd
    }
}

/// Inserts each entry as [`insert`](StateDict::insert) does.
impl<K: Into<String>> Extend<(K, Tensor)> for StateDict {
    fn extend<I: IntoIterator<Item = (K, Tensor)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl IntoIterator for StateDict {
    type Item = (String, Tensor);
    type IntoIter = std::vec::IntoIter<(String, Tensor)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a> IntoIterator for &'a StateDict {
    type Item = (&'a str, &'a Tensor);
    type IntoIter = StateDictIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// The entries of a [`StateDict`] by reference, in order: see [`StateDict::iter`].
#[derive(Clone, Debug)]
pub struct StateDictIter<'a>(std::slice::Iter<'a, (String, Tensor)>);

impl<'a> Iterator for StateDictIter<'a> {
    type Item = (&'a str, &'a Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (k.as_str(), v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for StateDictIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(k, v)| (k.as_str(), v))
    }
}

impl ExactSizeIterator for StateDictIter<'_> {}

impl std::iter::FusedIterator for StateDictIter<'_> {}

/// Lists each key with the shape and dtype of its tensor, not the values.
impl fmt::Debug for StateDict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Summary<'a>(&'a Tensor);

        impl fmt::Debug for Summary<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:?} {}", self.0.shape(), self.0.dtype())
            }
        }

        f.debug_map()
            .entries(self.iter().map(|(k, v)| (k, Summary(v))))
            .finish()
    }
}

/// The keys that [`Module::load_state_dict`](super::Module::load_state_dict) didn't match when it
/// succeeded, like PyTorch's `_IncompatibleKeys`. Both lists are empty after a strict load.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Keys of the module that the state dict has no entry for, in the module's order.
    pub missing_keys: Vec<String>,
    /// Keys of the state dict that the module doesn't have, in the state dict's order.
    pub unexpected_keys: Vec<String>,
}

/// Why [`Module::load_state_dict`](super::Module::load_state_dict) failed. Nothing was written.
///
/// `Display` formats it like PyTorch's error: missing keys, unexpected keys, then one line per
/// entry that can't be loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadError {
    /// Keys of the module that the state dict has no entry for; only filled by a strict load.
    pub missing_keys: Vec<String>,
    /// Keys of the state dict that the module doesn't have; only filled by a strict load.
    pub unexpected_keys: Vec<String>,
    /// One message per entry that can't be copied into the module: a shape mismatch, a device
    /// that can't be copied from or a target that can't be written, in the module's order.
    pub errors: Vec<String>,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let quoted = |keys: &[String]| {
            keys.iter()
                .map(|k| format!("\"{k}\""))
                .collect::<Vec<_>>()
                .join(", ")
        };
        f.write_str("Error(s) in loading state_dict:")?;
        if !self.missing_keys.is_empty() {
            let keys = quoted(&self.missing_keys);
            write!(f, "\n\tMissing key(s) in state_dict: {keys}. ")?;
        }
        if !self.unexpected_keys.is_empty() {
            let keys = quoted(&self.unexpected_keys);
            write!(f, "\n\tUnexpected key(s) in state_dict: {keys}. ")?;
        }
        for e in &self.errors {
            write!(f, "\n\t{e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for LoadError {}
