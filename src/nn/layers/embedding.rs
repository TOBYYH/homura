use std::fmt;

use super::Forward;
use crate::Tensor;
use crate::nn::{Module, Parameter, functional, init};

/// A lookup table of `num_embeddings` rows of size `embedding_dim`, PyTorch's `nn.Embedding`:
/// [`forward`](Embedding::forward) returns the rows at the given indices, with
/// [`functional::embedding`].
///
/// `weight` has the shape `[num_embeddings, embedding_dim]`, like PyTorch's, and is initialized
/// from `N(0, 1)` with the default dtype and device. With a [`padding_idx`](Embedding::padding_idx),
/// that row starts as zeros and never gets a gradient.
///
/// ```
/// use homura::nn::{Embedding, Module};
///
/// let table = Embedding::new(10, 4).padding_idx(0);
/// let rows = table.forward(&homura::tensor([[0i64, 3], [5, 0]]));
/// assert_eq!(rows.shape(), [2, 2, 4]);
/// assert_eq!(table.weight.select(0, 0).to_vec::<f32>(), [0.0; 4]);
/// assert_eq!(table.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
/// ```
#[derive(Module)]
#[non_exhaustive]
pub struct Embedding {
    /// The rows, `[num_embeddings, embedding_dim]`.
    pub weight: Parameter,
    /// The index whose row gets no gradient, if any (`0..num_embeddings`).
    pub padding_idx: Option<usize>,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    /// A table of `num_embeddings` rows of size `embedding_dim`
    /// (`nn.Embedding(num_embeddings, embedding_dim)`).
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Embedding {
        let table = Embedding {
            weight: Parameter::new(crate::zeros([num_embeddings, embedding_dim])),
            padding_idx: None,
            num_embeddings,
            embedding_dim,
        };
        table.reset_parameters();
        table
    }

    /// This table with `padding_idx` as the padding index (`padding_idx=...`), which may be
    /// negative to count from the end. Its row is set to zeros.
    ///
    /// # Panics
    ///
    /// Panics if `padding_idx` is not in `-num_embeddings..num_embeddings`, like PyTorch.
    #[must_use]
    pub fn padding_idx(mut self, padding_idx: i64) -> Embedding {
        let n = self.num_embeddings as i64;
        if padding_idx >= n || padding_idx < -n {
            panic!("Padding_idx must be within num_embeddings");
        }
        let index = if padding_idx < 0 {
            padding_idx + n
        } else {
            padding_idx
        };
        self.padding_idx = Some(index as usize);
        self.zero_padding_row();
        self
    }

    /// Initializes the weights again, like PyTorch: from `N(0, 1)`, with zeros in the padding
    /// row.
    pub fn reset_parameters(&self) {
        init::normal_(&self.weight, 0.0, 1.0);
        self.zero_padding_row();
    }

    fn zero_padding_row(&self) {
        if let Some(index) = self.padding_idx {
            let _no_grad = crate::no_grad();
            self.weight.select(0, index as i64).zero_();
        }
    }

    /// The rows at `input`, an `I64` or `I32` tensor of any shape; the result has the shape of
    /// `input` followed by `embedding_dim`.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        functional::embedding(input, &self.weight, self.padding_idx.map(|i| i as i64))
    }
}

impl Forward for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        Embedding::forward(self, input)
    }
}

/// The sizes, like PyTorch's `repr`:
/// `Embedding { num_embeddings: 10, embedding_dim: 4, padding_idx: Some(0), .. }`.
impl fmt::Debug for Embedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Embedding")
            .field("num_embeddings", &self.num_embeddings)
            .field("embedding_dim", &self.embedding_dim)
            .field("padding_idx", &self.padding_idx)
            .finish_non_exhaustive()
    }
}
