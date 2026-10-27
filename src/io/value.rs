//! [`Value`]: the Python objects of PyTorch checkpoints.

use std::collections::HashMap;
use std::fmt;

use crate::nn::{ModuleVersions, StateDict};
use crate::{Error, Result, Tensor};

/// A Python object of the kinds that state dicts and ordinary checkpoints hold: what
/// `torch.load(path, weights_only=True)` returns, for the objects homura supports.
///
/// Python integers outside the range of `i64`, `bytes`, sets, recursive structures and objects
/// of other classes are not supported.
///
/// ```
/// use homura::io::{Dict, Value};
/// use homura::nn::StateDict;
///
/// // {"model": state_dict, "epoch": 3, "losses": [0.5, 0.25]}
/// let mut state_dict = StateDict::new();
/// state_dict.insert("weight", homura::zeros([2]));
/// let checkpoint: Dict = [
///     ("model", Value::from(&state_dict)),
///     ("epoch", Value::from(3)),
///     ("losses", Value::list([0.5, 0.25])),
/// ]
/// .into_iter()
/// .collect();
/// let checkpoint = Value::from(checkpoint);
///
/// let Value::Dict(dict) = &checkpoint else { unreachable!() };
/// let model = StateDict::try_from(dict.get("model").unwrap())?;
/// assert_eq!(model["weight"].shape(), [2]);
/// # Ok::<(), homura::Error>(())
/// ```
#[derive(Clone, Debug)]
pub enum Value {
    /// `None`.
    None,
    /// `bool`.
    Bool(bool),
    /// `int`.
    Int(i64),
    /// `float`.
    Float(f64),
    /// `str`.
    Str(String),
    /// `list`.
    List(Vec<Value>),
    /// `tuple`.
    Tuple(Vec<Value>),
    /// `dict` or `collections.OrderedDict`.
    Dict(Dict),
    /// `torch.Tensor`, including `torch.nn.Parameter`.
    Tensor(Tensor),
}

impl Value {
    /// A `list` of the items.
    #[track_caller]
    pub fn list<T: Into<Value>>(items: impl IntoIterator<Item = T>) -> Value {
        Value::List(items.into_iter().map(Into::into).collect())
    }

    /// A `tuple` of the items.
    #[track_caller]
    pub fn tuple<T: Into<Value>>(items: impl IntoIterator<Item = T>) -> Value {
        Value::Tuple(items.into_iter().map(Into::into).collect())
    }

    /// The Python type's name, for messages.
    pub(super) fn type_name(&self) -> &'static str {
        match self {
            Value::None => "None",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Tuple(_) => "tuple",
            Value::Dict(d) if d.is_ordered() => "OrderedDict",
            Value::Dict(_) => "dict",
            Value::Tensor(_) => "Tensor",
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Value {
        Value::Int(i)
    }
}

/// So that integer literals without a suffix work: `Value::from(3)`.
impl From<i32> for Value {
    fn from(i: i32) -> Value {
        Value::Int(i.into())
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Value {
        Value::Float(f)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.to_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s)
    }
}

/// A `list`.
impl From<Vec<Value>> for Value {
    fn from(items: Vec<Value>) -> Value {
        Value::List(items)
    }
}

impl From<Dict> for Value {
    fn from(dict: Dict) -> Value {
        Value::Dict(dict)
    }
}

impl From<Tensor> for Value {
    fn from(t: Tensor) -> Value {
        Value::Tensor(t)
    }
}

/// The same tensor: a new handle, not a copy.
impl From<&Tensor> for Value {
    fn from(t: &Tensor) -> Value {
        Value::Tensor(t.clone())
    }
}

/// A key of a [`Dict`]: a Python `int` or `str`, the keys that checkpoints use (optimizer
/// state is keyed by parameter index).
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    /// `int`.
    Int(i64),
    /// `str`.
    Str(String),
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Key::Int(i) => write!(f, "{i}"),
            Key::Str(s) => write!(f, "{s:?}"),
        }
    }
}

impl From<i64> for Key {
    fn from(i: i64) -> Self {
        Key::Int(i)
    }
}

/// So that integer literals without a suffix work: `dict.get(0)`.
impl From<i32> for Key {
    fn from(i: i32) -> Self {
        Key::Int(i.into())
    }
}

impl From<&str> for Key {
    fn from(s: &str) -> Self {
        Key::Str(s.to_string())
    }
}

impl From<String> for Key {
    fn from(s: String) -> Self {
        Key::Str(s)
    }
}

/// A Python `dict` or `collections.OrderedDict`: values by [`Key`] in insertion order.
///
/// An ordered dict can also hold the module [versions](Dict::versions) that PyTorch keeps on a
/// state dict (its `_metadata` attribute), so a state dict inside a checkpoint keeps them.
/// Both kinds keep insertion order, as Python's do; the kind only tells which one a file had or
/// will have.
#[derive(Clone, Default)]
pub struct Dict {
    items: Vec<(Key, Value)>,
    index: HashMap<Key, usize>,
    ordered: bool,
    versions: Option<ModuleVersions>,
}

impl Dict {
    /// An empty `dict`.
    #[track_caller]
    pub fn new() -> Dict {
        Dict::default()
    }

    /// An empty `collections.OrderedDict`, without versions.
    #[track_caller]
    pub fn new_ordered() -> Dict {
        Dict {
            ordered: true,
            ..Dict::default()
        }
    }

    /// Whether it is an `OrderedDict` rather than a `dict`.
    #[track_caller]
    pub fn is_ordered(&self) -> bool {
        self.ordered
    }

    /// The number of entries.
    #[track_caller]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether there are no entries.
    #[track_caller]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The value for `key`.
    #[track_caller]
    pub fn get(&self, key: impl Into<Key>) -> Option<&Value> {
        self.index.get(&key.into()).map(|&i| &self.items[i].1)
    }

    /// Sets the value for `key` and returns the one it had. A new key goes last; an existing
    /// one keeps its position, as in Python.
    #[track_caller]
    pub fn insert(&mut self, key: impl Into<Key>, value: impl Into<Value>) -> Option<Value> {
        let (key, value) = (key.into(), value.into());
        match self.index.get(&key) {
            Some(&i) => Some(std::mem::replace(&mut self.items[i].1, value)),
            None => {
                self.index.insert(key.clone(), self.items.len());
                self.items.push((key, value));
                None
            }
        }
    }

    /// The entries, in order.
    pub(super) fn items(&self) -> &[(Key, Value)] {
        &self.items
    }

    /// The entries, in order.
    #[track_caller]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&Key, &Value)> + ExactSizeIterator {
        self.items.iter().map(|(k, v)| (k, v))
    }

    /// The versions of the modules of the state dict this dict holds, if it has any (PyTorch's
    /// `state_dict._metadata`). Only an `OrderedDict` can have them.
    #[track_caller]
    pub fn versions(&self) -> Option<&ModuleVersions> {
        self.versions.as_ref()
    }

    /// Sets or removes the versions.
    ///
    /// # Panics
    ///
    /// If `versions` is `Some` and this is not an `OrderedDict`: a Python `dict` can't have
    /// attributes.
    #[track_caller]
    pub fn set_versions(&mut self, versions: Option<ModuleVersions>) {
        assert!(
            versions.is_none() || self.ordered,
            "only an OrderedDict can have versions; create it with Dict::new_ordered"
        );
        self.versions = versions;
    }
}

/// A `dict` of the entries, inserted in order as [`insert`](Dict::insert) does.
impl<K: Into<Key>, V: Into<Value>> FromIterator<(K, V)> for Dict {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Dict {
        let mut dict = Dict::new();
        dict.extend(iter);
        dict
    }
}

/// Inserts each entry as [`insert`](Dict::insert) does.
impl<K: Into<Key>, V: Into<Value>> Extend<(K, V)> for Dict {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl fmt::Debug for Dict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ordered {
            f.write_str("OrderedDict")?;
            if let Some(versions) = &self.versions {
                write!(f, "(versions: {versions:?}) ")?;
            } else {
                f.write_str(" ")?;
            }
        }
        f.debug_map().entries(self.iter()).finish()
    }
}

/// An `OrderedDict` of the tensors by key with the state dict's versions, as PyTorch saves a
/// state dict. The tensors are shared, not copied.
impl From<&StateDict> for Value {
    fn from(state_dict: &StateDict) -> Value {
        let mut dict = Dict::new_ordered();
        for (key, tensor) in state_dict {
            dict.insert(key, Value::Tensor(tensor.clone()));
        }
        dict.set_versions(state_dict.versions().cloned());
        Value::Dict(dict)
    }
}

/// A `dict` or `OrderedDict` of tensors by string key, in order. The versions of an
/// `OrderedDict` become the state dict's; a `dict` gives a state dict without versions. The
/// tensors are shared, not copied.
///
/// # Errors
///
/// [`Error::Format`] if the value is not a dict, a key is not a string or a value is not a
/// tensor.
impl TryFrom<&Value> for StateDict {
    type Error = Error;

    fn try_from(value: &Value) -> Result<StateDict> {
        let Value::Dict(dict) = value else {
            return Err(Error::Format(format!(
                "expected a state dict (a dict of tensors), found a {}",
                value.type_name()
            )));
        };
        let mut state_dict = StateDict::new();
        for (key, value) in dict.iter() {
            let Key::Str(key) = key else {
                return Err(Error::Format(format!(
                    "expected a state dict (a dict of tensors), found the key {key:?}, which is \
                     not a string"
                )));
            };
            let Value::Tensor(tensor) = value else {
                return Err(Error::Format(format!(
                    "expected a state dict (a dict of tensors), found a {} for {key:?}",
                    value.type_name()
                )));
            };
            state_dict.insert(key.clone(), tensor.clone());
        }
        *state_dict.versions_mut() = dict.versions().cloned();
        Ok(state_dict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dicts_keep_insertion_order_and_positions() {
        let mut dict = Dict::new();
        assert!(dict.insert("b", Value::Int(1)).is_none());
        assert!(dict.insert(0, Value::None).is_none());
        assert!(dict.insert("a", Value::Bool(true)).is_none());
        assert!(matches!(
            dict.insert("b", Value::Int(2)),
            Some(Value::Int(1))
        ));
        let keys: Vec<&Key> = dict.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, [&Key::from("b"), &Key::Int(0), &Key::from("a")]);
        assert!(matches!(dict.get("b"), Some(Value::Int(2))));
        assert!(matches!(dict.get(0i64), Some(Value::None)));
        // An int key and a str key with the same digits are different keys.
        assert!(dict.get("0").is_none());
        assert_eq!(dict.len(), 3);
        assert!(!dict.is_ordered() && Dict::new_ordered().is_ordered());
        assert_eq!(
            format!("{dict:?}"),
            r#"{"b": Int(2), 0: None, "a": Bool(true)}"#
        );
    }
}
