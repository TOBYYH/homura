//! The activation functions that model configurations name (`"hidden_act": "relu"`),
//! transformers' `ACT2FN`, as layers of homura's core.

use homura::Tensor;
use homura::nn::{Forward, Gelu, Identity, Member, Module, Relu, Sigmoid, Silu, Tanh, Visitor};

/// An activation layer chosen by name, as transformers' `ACT2FN[name]` is, or [`Identity`]
/// where a model has none (transformers' `nn.Identity()` for `activation=None`).
///
/// As a module it is the layer it holds: its [`type_name`](Module::type_name) and settings are
/// the layer's, so a model prints like transformers'.
///
/// ```
/// use homura::nn::Module;
/// use homura_models::activations::Activation;
///
/// let relu = Activation::from_name("relu").unwrap();
/// assert_eq!(relu.forward(&homura::tensor([-1.0f32, 2.0])).to_vec::<f32>(), [0.0, 2.0]);
/// assert_eq!(relu.display().to_string(), "Relu()");
/// assert!(Activation::from_name("mish").is_none());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Activation {
    /// `"relu"`: `nn.ReLU`.
    Relu(Relu),
    /// `"sigmoid"`: `nn.Sigmoid`.
    Sigmoid(Sigmoid),
    /// `"tanh"`: `nn.Tanh`.
    Tanh(Tanh),
    /// `"silu"` and `"swish"`: transformers' `SiLUActivation` and `nn.SiLU`.
    Silu(Silu),
    /// `"gelu"` (exact) and `"gelu_pytorch_tanh"` (the tanh approximation): transformers'
    /// `GELUActivation` and `GELUTanh`.
    Gelu(Gelu),
    /// No activation.
    Identity(Identity),
}

impl Activation {
    /// The names that [`from_name`](Activation::from_name) knows.
    pub const NAMES: [&'static str; 7] = [
        "relu",
        "sigmoid",
        "tanh",
        "silu",
        "swish",
        "gelu",
        "gelu_pytorch_tanh",
    ];

    /// The activation that transformers' `ACT2FN[name]` gives, among [`NAMES`](Self::NAMES), or
    /// `None` for other names, also for those of transformers' other activations, which homura
    /// doesn't have. The functions are the same as transformers' for every input; transformers
    /// prints `silu`, `gelu` and `gelu_pytorch_tanh` as its own classes (`SiLUActivation()`,
    /// `GELUActivation()`, `GELUTanh()`), homura as the core layers (`Silu()`, `Gelu(..)`).
    #[track_caller]
    pub fn from_name(name: &str) -> Option<Activation> {
        Some(match name {
            "relu" => Activation::Relu(Relu),
            "sigmoid" => Activation::Sigmoid(Sigmoid),
            "tanh" => Activation::Tanh(Tanh),
            "silu" | "swish" => Activation::Silu(Silu),
            "gelu" => Activation::Gelu(Gelu::new()),
            "gelu_pytorch_tanh" => {
                Activation::Gelu(Gelu::new().approximate(homura::GeluApproximate::Tanh))
            }
            _ => return None,
        })
    }

    /// The activation of `input`: a new tensor, or `input` itself for [`Identity`].
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Activation::Relu(layer) => layer.forward(input),
            Activation::Sigmoid(layer) => layer.forward(input),
            Activation::Tanh(layer) => layer.forward(input),
            Activation::Silu(layer) => layer.forward(input),
            Activation::Gelu(layer) => layer.forward(input),
            Activation::Identity(layer) => layer.forward(input),
        }
    }

    fn layer(&self) -> &dyn Module {
        match self {
            Activation::Relu(layer) => layer,
            Activation::Sigmoid(layer) => layer,
            Activation::Tanh(layer) => layer,
            Activation::Silu(layer) => layer,
            Activation::Gelu(layer) => layer,
            Activation::Identity(layer) => layer,
        }
    }
}

impl Forward for Activation {
    fn forward(&self, input: &Tensor) -> Tensor {
        Activation::forward(self, input)
    }
}

impl Module for Activation {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        self.layer().visit(v);
    }

    fn extra_repr(&self) -> String {
        self.layer().extra_repr()
    }

    fn type_name(&self) -> &'static str {
        self.layer().type_name()
    }

    fn version(&self) -> u32 {
        self.layer().version()
    }

    fn added_in(&self, name: &str) -> Option<u32> {
        self.layer().added_in(name)
    }
}

impl Member for Activation {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}
