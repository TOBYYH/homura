use std::fmt;

use super::Forward;
use crate::Tensor;
use crate::nn::{Member, Module, Visitor};

/// Layers applied one after another (`nn.Sequential`): the output of each is the input of the
/// next. The layers are its submodules, named `0`, `1`, ..., so their parameters have the keys
/// of PyTorch's `nn.Sequential`: `0.weight`.
///
/// ```
/// use homura::Tensor;
/// use homura::nn::{Forward, Module, Parameter, Sequential};
///
/// #[derive(Module)]
/// struct Scale {
///     weight: Parameter,
/// }
///
/// impl Forward for Scale {
///     fn forward(&self, input: &Tensor) -> Tensor {
///         input * &*self.weight
///     }
/// }
///
/// let scale = |w: f32| Scale { weight: Parameter::new(homura::tensor([w])) };
/// let model = Sequential::new()
///     .append(scale(2.0))
///     .append(Sequential::new().append(scale(3.0)));
/// let keys: Vec<_> = model.state_dict().keys().map(String::from).collect();
/// assert_eq!(keys, ["0.weight", "1.0.weight"]);
/// assert_eq!(model.forward(&homura::tensor([1.0f32])).to_vec::<f32>(), [6.0]);
/// ```
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Forward + Send + Sync>>,
}

impl Sequential {
    /// A `Sequential` without layers, whose forward computation returns its input.
    pub fn new() -> Sequential {
        Sequential::default()
    }

    /// This `Sequential` with `layer` appended (`append(layer)`, which also returns the
    /// `Sequential`).
    #[must_use]
    pub fn append(mut self, layer: impl Forward + Send + Sync + 'static) -> Sequential {
        self.push(layer);
        self
    }

    /// Appends `layer`.
    pub fn push(&mut self, layer: impl Forward + Send + Sync + 'static) {
        self.layers.push(Box::new(layer));
    }

    /// The number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether there are no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The layer at `index`, if there is one.
    pub fn get(&self, index: usize) -> Option<&(dyn Forward + Send + Sync)> {
        self.layers.get(index).map(|layer| &**layer)
    }

    /// The layers, in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &(dyn Forward + Send + Sync)> {
        self.layers.iter().map(|layer| &**layer)
    }

    /// Applies the layers to `input` in order; without layers, returns `input` (the same tensor).
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }
}

impl Forward for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        Sequential::forward(self, input)
    }
}

impl Module for Sequential {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        for (i, layer) in self.layers.iter().enumerate() {
            layer.visit_as(&i.to_string(), v);
        }
    }
}

impl Member for Sequential {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}

impl fmt::Debug for Sequential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sequential")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}
