# homura

A tensor computing framework written in Rust that aims to feel as close to PyTorch as possible.
Each hardware backend is built directly on the vendor's latest official development tools, so that
it can get the best performance out of the hardware.

The first GPU backend targets AMD ROCm. Backends for other hardware will be added once the ROCm
backend is stable.

> This project is in early development: only the CPU reference backend and the meta device are
> implemented so far. The API is not stable yet and it is not ready for real-world use.

## Status

Implemented so far:

- `Tensor` with PyTorch's reference semantics, dtypes and type promotion, views, indexing,
  broadcasting and forward operators, on a CPU reference backend and on the meta device, which
  computes only shapes and dtypes.
- Advanced indexing with integer tensors and bool masks, `index_put` / `index_put_`, constant
  padding with `nn::functional::pad`, and window views with `Tensor::unfold`.
- Automatic differentiation like PyTorch's autograd: backward passes, gradients of gradients to
  any order, in-place operations checked with version counters, `retain_grad`, and
  `gradcheck` / `gradgradcheck`. Gradients are checked against PyTorch, including for a Llama
  decoder layer.
- Custom differentiable operations through `autograd::Function`, including higher-order
  derivatives through saved inputs and outputs.
- Model files: `homura::load` / `homura::save` for state dicts in safetensors files, read
  straight into tensor storage; safetensors `__metadata__`; sharded checkpoints in the Hugging
  Face layout (`model.safetensors.index.json`) and zero-copy loading by memory mapping
  (`unsafe fn io::load_mmap`) through `homura::io`; `io::Value` for the Python objects of
  checkpoints. PyTorch `.pt` files (including `pytorch_model.bin` and its sharded index) load
  with their tensors through a restricted unpickler that runs no code, and `homura::save` writes
  them as `torch.save` does; `io::load_value` and `io::save_value` read and write whole
  checkpoints.
- The core of `nn` modules: `Parameter`, `#[derive(Module)]` and the `Module` trait with
  PyTorch's traversals, `state_dict` / `load_state_dict` and in-place conversion with `to`;
  `Sequential`; `nn::init`, `Linear`, `Embedding`, `LayerNorm`, `RmsNorm`, `Dropout`, the
  activation layers and the CNN layers `Conv2d`, `BatchNorm2d`, `MaxPool2d`,
  `AdaptiveAvgPool2d` and `Flatten`, initialized like PyTorch's, and `Identity`; the losses `mse_loss`,
  `nll_loss`, `cross_entropy` and `binary_cross_entropy_with_logits`; the functions `conv2d`,
  `max_pool2d`, `adaptive_avg_pool2d` and `batch_norm`.
- The optimizers `optim::Sgd`, `Adam` and `AdamW`, with parameter groups, checked step by step
  against PyTorch's. Ten steps of training a small MLP, and a small CNN with batch
  normalization, give PyTorch's losses, gradients, parameters and optimizer state (and the
  batch norm statistics) at every step. Their state dicts are PyTorch's, so training resumes
  from a checkpoint written by either library.

Not available yet: GPU backends (ROCm comes first), most neural network layers, autograd hooks,
and in-place operations on views that require grad.

## Example

```rust
use homura::autograd::{self, GradOptions};

let x = homura::tensor([1.0, 2.0, 3.0]);
x.requires_grad_(true);

// y = sum(x³), so dy/dx = 3x²
let y = (&x * &x * &x).sum();
y.backward();
assert_eq!(x.grad().unwrap().to_vec::<f32>(), [3.0, 12.0, 27.0]);

// Keep the graph of the gradient to differentiate it again: d(sum(3x²))/dx = 6x
let y = (&x * &x * &x).sum();
let options = GradOptions { create_graph: true, ..Default::default() };
let [dy] = autograd::grad_array(&[&y], [&x], options).map(Option::unwrap);
let [d2y] = autograd::grad_array(&[&dy.sum()], [&x], GradOptions::default()).map(Option::unwrap);
assert_eq!(d2y.to_vec::<f32>(), [6.0, 12.0, 18.0]);
```

## Models from Hugging Face

The workspace crate [homura-models](homura-models) has models whose module trees, parameter names
and configuration files are those of transformers, so that Hugging Face repositories load as they
are, from a local directory or the Hugging Face cache (`hf download microsoft/resnet-50`). The
example below is compiled as a test of homura-models:

```rust,ignore
use homura_models::hub::{Repo, Revision};
use homura_models::pretrained::{FromPretrained, PretrainedOptions};
use homura_models::resnet::ResNetForImageClassification;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = Repo::cached("microsoft/resnet-50", Revision::main())?;
    let model = ResNetForImageClassification::from_pretrained(&repo, PretrainedOptions::default())?;
    // Images resized, cropped and normalized as the repository's preprocessor_config.json says.
    let pixel_values = homura::zeros([1, 3, 224, 224]);
    let logits = homura::with_no_grad(|| model.forward(&pixel_values));
    assert_eq!(logits.shape(), [1, 1000]);
    Ok(())
}
```

## Minimum supported Rust version

Rust 1.88 or later. Raising it is a minor version change.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
