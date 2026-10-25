//! [`StateDict`] and the results of [`Module::load_state_dict`](super::Module::load_state_dict).

use std::collections::HashMap;
use std::fmt;

use crate::Tensor;

/// Values by key in insertion order, with an index for lookups: the entries of a [`StateDict`]
/// and of [`ModuleVersions`].
#[derive(Clone)]
struct Ordered<V> {
    items: Vec<(String, V)>,
    index: HashMap<String, usize>,
}

impl<V> Default for Ordered<V> {
    fn default() -> Self {
        Ordered {
            items: vec![],
            index: HashMap::new(),
        }
    }
}

impl<V> Ordered<V> {
    fn get(&self, key: &str) -> Option<&V> {
        self.index.get(key).map(|&i| &self.items[i].1)
    }

    /// A new key goes last; an existing one keeps its position.
    fn insert(&mut self, key: String, value: V) -> Option<V> {
        match self.index.get(&key) {
            Some(&i) => Some(std::mem::replace(&mut self.items[i].1, value)),
            None => {
                self.index.insert(key.clone(), self.items.len());
                self.items.push((key, value));
                None
            }
        }
    }

    fn remove(&mut self, key: &str) -> Option<V> {
        let i = self.index.remove(key)?;
        let (_, value) = self.items.remove(i);
        for position in self.index.values_mut() {
            if *position > i {
                *position -= 1;
            }
        }
        Some(value)
    }

    /// Gives the keys that `rename` maps to `Some` their new key, one after another, as Python's
    /// `d[new] = d.pop(key)` does: the renamed entry goes last, or replaces the value of an
    /// existing entry with its new key in that entry's position.
    fn rename(&mut self, mut rename: impl FnMut(&str) -> Option<String>) {
        let renames: Vec<(String, String)> = self
            .items
            .iter()
            .filter_map(|(key, _)| Some((key.clone(), rename(key)?)))
            .collect();
        for (key, new) in renames {
            if let Some(value) = self.remove(&key) {
                self.insert(new, value);
            }
        }
    }
}

/// The parameters and persistent buffers of a module by name, in the module's order, like the
/// `OrderedDict` that PyTorch's `state_dict()` returns, and the versions of its modules (PyTorch's
/// `state_dict._metadata`, see [`versions`](StateDict::versions)).
///
/// [`Module::state_dict`](super::Module::state_dict) fills it, [`homura::load`](crate::load)
/// reads it from a file and [`homura::save`](crate::save) writes it. It can also be collected
/// from `(String, Tensor)` pairs, such as those [`load_safetensors`](crate::load_safetensors)
/// returns, and iterated by reference to save it with
/// [`save_safetensors`](crate::save_safetensors). State dicts read from safetensors files have no
/// versions: the format doesn't store them, as in PyTorch.
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
    entries: Ordered<Tensor>,
    versions: Option<ModuleVersions>,
}

impl StateDict {
    /// An empty state dict, without versions.
    #[track_caller]
    pub fn new() -> StateDict {
        StateDict::default()
    }

    /// The number of entries.
    #[track_caller]
    pub fn len(&self) -> usize {
        self.entries.items.len()
    }

    /// Whether there are no entries.
    #[track_caller]
    pub fn is_empty(&self) -> bool {
        self.entries.items.is_empty()
    }

    /// The tensor stored under `key`.
    #[track_caller]
    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.entries.get(key)
    }

    /// Whether there is an entry for `key`.
    #[track_caller]
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.index.contains_key(key)
    }

    /// Stores `value` under `key` and returns the tensor stored there before, if any. A new key
    /// goes last; an existing one keeps its position.
    #[track_caller]
    pub fn insert(&mut self, key: impl Into<String>, value: Tensor) -> Option<Tensor> {
        self.entries.insert(key.into(), value)
    }

    /// Removes the entry for `key` and returns its tensor, keeping the order of the others.
    #[track_caller]
    pub fn remove(&mut self, key: &str) -> Option<Tensor> {
        self.entries.remove(key)
    }

    /// The keys, in order.
    #[track_caller]
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.items.iter().map(|(k, _)| k.as_str())
    }

    /// The tensors, in order.
    #[track_caller]
    pub fn values(&self) -> impl Iterator<Item = &Tensor> {
        self.entries.items.iter().map(|(_, v)| v)
    }

    /// The entries, in order.
    #[track_caller]
    pub fn iter(&self) -> StateDictIter<'_> {
        StateDictIter(self.entries.items.iter())
    }

    /// The versions of the modules the entries come from, PyTorch's `state_dict._metadata`, or
    /// `None` if there are none.
    ///
    /// [`Module::state_dict`](super::Module::state_dict) records the
    /// [`version`](super::Module::version) of every module of the tree by its path, the root
    /// under `""`. Loading reads them to accept old state dicts without the persistent buffers a
    /// module added in a later version (see [`Module::added_in`](super::Module::added_in)), which
    /// only [`BatchNorm2d`](super::BatchNorm2d) has among homura's layers. A state dict without
    /// versions counts as old, as a plain `dict` does in PyTorch.
    ///
    /// ```
    /// use homura::nn::{BatchNorm2d, Linear, Module, Sequential};
    ///
    /// let model = Sequential::new().append(Linear::new(2, 2)).append(BatchNorm2d::new(2));
    /// let sd = model.state_dict();
    /// let versions: Vec<(&str, u32)> = sd.versions().unwrap().iter().collect();
    /// assert_eq!(versions, [("", 1), ("0", 1), ("1", 2)]);
    /// assert_eq!(sd.module_version("1"), Some(2));
    /// ```
    #[track_caller]
    pub fn versions(&self) -> Option<&ModuleVersions> {
        self.versions.as_ref()
    }

    /// The versions, to change, add (`Some`) or remove (`None`) them: see
    /// [`versions`](StateDict::versions).
    #[track_caller]
    pub fn versions_mut(&mut self) -> &mut Option<ModuleVersions> {
        &mut self.versions
    }

    /// The version of the module with the path `path` (`""` for the root), if there is one.
    #[track_caller]
    pub fn module_version(&self, path: &str) -> Option<u32> {
        self.versions.as_ref()?.get(path)
    }

    /// Sets the version of the module with the path `path`, adding versions if there are none,
    /// and returns the version it had before.
    #[track_caller]
    pub fn set_module_version(&mut self, path: impl Into<String>, version: u32) -> Option<u32> {
        self.versions
            .get_or_insert_with(ModuleVersions::new)
            .insert(path, version)
    }

    /// Removes `prefix` from the keys that start with it, PyTorch's
    /// `consume_prefix_in_state_dict_if_present`: to load the state dict of a wrapped model, such
    /// as one saved from `DistributedDataParallel` with keys `module.*`. Each renamed entry goes
    /// last, unless its new key already exists: then it replaces that entry's tensor in its
    /// position. The versions are renamed the same way, and the version of the module whose path
    /// is `prefix` without its final `.` becomes the root's (`""`).
    ///
    /// ```
    /// use homura::nn::{Linear, Module, StateDict};
    ///
    /// let mut sd = StateDict::new();
    /// sd.insert("module.weight", homura::zeros([2, 3]));
    /// sd.insert("module.bias", homura::zeros([2]));
    /// sd.consume_prefix("module.");
    /// Linear::new(3, 2).load_state_dict(&sd).unwrap();
    /// ```
    ///
    /// PyTorch compares the paths of the versions with `prefix` without any `.`, which differs
    /// for a prefix with a `.` inside (`a.b.`): it doesn't rename the version of `a.b` and renames
    /// that of a module named `ab` instead.
    #[track_caller]
    pub fn consume_prefix(&mut self, prefix: &str) {
        self.entries
            .rename(|key| Some(key.strip_prefix(prefix)?.to_string()));
        if let Some(versions) = &mut self.versions {
            let own = prefix.strip_suffix('.').unwrap_or(prefix);
            versions.0.rename(|path| match path.strip_prefix(prefix) {
                _ if path.is_empty() => None,
                Some(rest) => Some(rest.to_string()),
                None => (path == own).then(String::new),
            });
        }
    }

    /// Adds `prefix` (normally ending with `.`) to every key, as PyTorch's
    /// `state_dict(prefix=...)` does: to load into a module that holds the model as a submodule.
    /// The versions get the prefix too, the root's under `prefix` without its final `.`.
    ///
    /// ```
    /// use homura::nn::{Linear, Module};
    ///
    /// let mut sd = Linear::new(3, 2).state_dict();
    /// sd.add_prefix("encoder.");
    /// assert_eq!(sd.keys().collect::<Vec<_>>(), ["encoder.weight", "encoder.bias"]);
    /// assert_eq!(sd.versions().unwrap().iter().collect::<Vec<_>>(), [("encoder", 1)]);
    /// ```
    #[track_caller]
    pub fn add_prefix(&mut self, prefix: &str) {
        let entries = std::mem::take(&mut self.entries);
        for (key, value) in entries.items {
            self.entries.insert(format!("{prefix}{key}"), value);
        }
        if let Some(versions) = &mut self.versions {
            let own = prefix.strip_suffix('.').unwrap_or(prefix);
            *versions = std::mem::take(versions)
                .into_iter()
                .map(|(path, version)| match path.as_str() {
                    "" => (own.to_string(), version),
                    _ => (format!("{prefix}{path}"), version),
                })
                .collect();
        }
    }

    /// The entries whose keys start with `prefix` (normally the path of a submodule followed by
    /// `.`), without it: the state dict of that submodule, such as
    /// `model.layer4.load_state_dict(&sd.sub("layer4."))`. The tensors are shared with this state
    /// dict. Of the versions, the submodule's own (its path is `prefix` without the final `.`)
    /// becomes the root's (`""`) and those of its descendants lose the prefix; the others are
    /// left out.
    ///
    /// ```
    /// use homura::nn::{BatchNorm2d, Linear, Module, Sequential};
    ///
    /// let model = Sequential::new().append(Linear::new(2, 2)).append(BatchNorm2d::new(2));
    /// let sd = model.state_dict().sub("1.");
    /// assert_eq!(sd.len(), 5);
    /// assert_eq!(sd.module_version(""), Some(2));
    /// BatchNorm2d::new(2).load_state_dict(&sd).unwrap();
    /// ```
    #[track_caller]
    pub fn sub(&self, prefix: &str) -> StateDict {
        let mut sub: StateDict = self
            .iter()
            .filter_map(|(key, value)| Some((key.strip_prefix(prefix)?, value.clone())))
            .collect();
        sub.versions = self.versions.as_ref().map(|versions| {
            let own = prefix.strip_suffix('.').unwrap_or(prefix);
            versions
                .iter()
                .filter_map(|(path, version)| match path.strip_prefix(prefix) {
                    _ if path == own => Some(("", version)),
                    Some(rest) => Some((rest, version)),
                    None => None,
                })
                .collect()
        });
        sub
    }

    /// The state dict with each key renamed by `f`, in the same order; the entries for which `f`
    /// returns `None` are left out. Keys can be renamed in any way, so the module paths of the
    /// versions can't be known: the result has no versions (set them with
    /// [`set_module_version`](StateDict::set_module_version) if loading needs them).
    ///
    /// ```
    /// use homura::nn::{Linear, Module};
    ///
    /// let sd = Linear::new(3, 2).state_dict();
    /// let sd = sd.map_keys(|key| (key != "bias").then(|| format!("proj.{key}")));
    /// assert_eq!(sd.keys().collect::<Vec<_>>(), ["proj.weight"]);
    /// assert!(sd.versions().is_none());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `f` gives two entries the same key, after it has been called for every entry.
    #[track_caller]
    pub fn map_keys(self, mut f: impl FnMut(&str) -> Option<String>) -> StateDict {
        let mut out = StateDict::new();
        // The first key given twice, with the keys it was given to.
        let mut duplicate = None;
        let mut sources: HashMap<String, String> = HashMap::new();
        for (key, value) in self.entries.items {
            let Some(new) = f(&key) else {
                continue;
            };
            if let Some(first) = sources.get(&new) {
                duplicate.get_or_insert_with(|| (new, first.clone(), key));
                continue;
            }
            sources.insert(new.clone(), key);
            out.insert(new, value);
        }
        if let Some((new, first, second)) = duplicate {
            panic!("map_keys gave the key {new:?} to both {first:?} and {second:?}");
        }
        out
    }
}

/// The tensor stored under a key.
///
/// # Panics
///
/// Panics if there is no entry for the key.
impl std::ops::Index<&str> for StateDict {
    type Output = Tensor;

    #[track_caller]
    fn index(&self, key: &str) -> &Tensor {
        match self.get(key) {
            Some(t) => t,
            None => panic!("no entry {key:?} in the state dict"),
        }
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
        self.entries.items.into_iter()
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

/// The versions of the modules of a state dict by module path (`""` for the root), in order:
/// PyTorch's `state_dict._metadata`, which has `{"version": n}` for each path. See
/// [`StateDict::versions`].
///
/// ```
/// use homura::nn::ModuleVersions;
///
/// let mut versions = ModuleVersions::new();
/// versions.insert("", 1);
/// versions.insert("bn", 2);
/// assert_eq!(versions.get("bn"), Some(2));
/// assert_eq!(versions.iter().collect::<Vec<_>>(), [("", 1), ("bn", 2)]);
/// ```
#[derive(Clone, Default)]
pub struct ModuleVersions(Ordered<u32>);

impl ModuleVersions {
    /// No versions.
    #[track_caller]
    pub fn new() -> ModuleVersions {
        ModuleVersions::default()
    }

    /// The number of module paths.
    #[track_caller]
    pub fn len(&self) -> usize {
        self.0.items.len()
    }

    /// Whether there are no module paths.
    #[track_caller]
    pub fn is_empty(&self) -> bool {
        self.0.items.is_empty()
    }

    /// The version of the module with the path `path`.
    #[track_caller]
    pub fn get(&self, path: &str) -> Option<u32> {
        self.0.get(path).copied()
    }

    /// Sets the version of the module with the path `path` and returns the one it had. A new
    /// path goes last; an existing one keeps its position.
    #[track_caller]
    pub fn insert(&mut self, path: impl Into<String>, version: u32) -> Option<u32> {
        self.0.insert(path.into(), version)
    }

    /// Removes the version of the module with the path `path` and returns it, keeping the order
    /// of the others.
    #[track_caller]
    pub fn remove(&mut self, path: &str) -> Option<u32> {
        self.0.remove(path)
    }

    /// The module paths with their versions, in order.
    #[track_caller]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&str, u32)> + ExactSizeIterator {
        self.0.items.iter().map(|(path, v)| (path.as_str(), *v))
    }
}

/// Equal if they have the same paths with the same versions in the same order.
impl PartialEq for ModuleVersions {
    fn eq(&self, other: &Self) -> bool {
        self.0.items == other.0.items
    }
}

impl Eq for ModuleVersions {}

impl<K: Into<String>> FromIterator<(K, u32)> for ModuleVersions {
    fn from_iter<I: IntoIterator<Item = (K, u32)>>(iter: I) -> Self {
        let mut versions = ModuleVersions::new();
        versions.extend(iter);
        versions
    }
}

/// Inserts each version as [`insert`](ModuleVersions::insert) does.
impl<K: Into<String>> Extend<(K, u32)> for ModuleVersions {
    fn extend<I: IntoIterator<Item = (K, u32)>>(&mut self, iter: I) {
        for (path, version) in iter {
            self.insert(path, version);
        }
    }
}

impl IntoIterator for ModuleVersions {
    type Item = (String, u32);
    type IntoIter = std::vec::IntoIter<(String, u32)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.items.into_iter()
    }
}

impl fmt::Debug for ModuleVersions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Options of [`Module::load_state_dict_with`](super::Module::load_state_dict_with), built
/// from the default with setters: `LoadOptions::default().strict(false)`.
///
/// ```
/// use homura::nn::LoadOptions;
///
/// assert!(LoadOptions::default().is_strict());
/// assert!(!LoadOptions::default().strict(false).is_strict());
/// ```
///
/// More options will be added, so the fields are private and it can't be built with a struct
/// expression from another crate:
///
/// ```compile_fail,E0639
/// use homura::nn::LoadOptions;
///
/// let options = LoadOptions { strict: false, ..Default::default() };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LoadOptions {
    strict: bool,
    assign: bool,
}

impl Default for LoadOptions {
    /// Strict and copying, like [`Module::load_state_dict`](super::Module::load_state_dict).
    fn default() -> Self {
        LoadOptions {
            strict: true,
            assign: false,
        }
    }
}

impl LoadOptions {
    /// Whether missing and unexpected keys are errors (`strict`, true by default). If not, the
    /// [`LoadReport`] lists them.
    #[must_use]
    #[track_caller]
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// The value set by [`strict`](LoadOptions::strict).
    #[track_caller]
    pub fn is_strict(&self) -> bool {
        self.strict
    }

    /// Whether the parameters and buffers take the data of the entries instead of copying it
    /// (`assign`, false by default). With assignment, a parameter gets the dtype, device and
    /// layout of its entry and shares its storage, without a copy: the way to load into a model
    /// built on the meta device (see [`with_device`](crate::with_device)).
    ///
    /// Unlike PyTorch, which replaces them with new parameters and buffers, homura keeps them:
    /// handles taken before (an optimizer's, a shared parameter's other names) see the new data,
    /// and a parameter keeps whether it requires grad. Its `.grad` is cleared. Graphs and views
    /// created from the old data can't be used any more (their backward passes and operations
    /// panic). An optimizer's state for the parameters stays as it was: after a change of dtype
    /// or device, load the optimizer's state too or create it again.
    ///
    /// ```
    /// use homura::nn::{LoadOptions, Linear, Module};
    /// use homura::{DType, Device};
    ///
    /// let trained = Linear::new(4, 2);
    /// trained.to(DType::F64);
    /// let model = homura::with_device(Device::Meta, || Linear::new(4, 2));
    /// let options = LoadOptions::default().assign(true);
    /// model.load_state_dict_with(&trained.state_dict(), options).unwrap();
    /// assert_eq!((model.weight.dtype(), model.weight.device()), (DType::F64, Device::Cpu));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn assign(mut self, assign: bool) -> Self {
        self.assign = assign;
        self
    }

    /// The value set by [`assign`](LoadOptions::assign).
    #[track_caller]
    pub fn is_assign(&self) -> bool {
        self.assign
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
    /// One message per entry that can't be loaded into the module: a shape mismatch, a device
    /// that can't be copied from or a target that can't be written; with assignment, a target
    /// that is a view or has a history, or a parameter that requires grad and an entry that isn't
    /// floating point. In the module's order.
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
