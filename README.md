# homura

A tensor computing framework written in Rust that aims to feel as close to PyTorch as possible.
Each hardware backend is built directly on the vendor's latest official development tools, so that
it can get the best performance out of the hardware. All compute kernels are homura's own: GPU
backends do not call vendor compute libraries such as hipBLASLt, rocBLAS or MIOpen. Performance work
is ongoing and gives priority to the tensor shapes that are common in machine learning.

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

- The start of the ROCm backend, behind the `rocm` feature: building homura's ROCm layer,
  querying devices (`Device::rocm_count`, `properties`, `mem_get_info`, `synchronize`) and
  device memory from a caching allocator (`memory_stats`, `reset_peak_memory_stats`,
  `empty_cache`, which the CPU reports too). Tensors move to and from a ROCm GPU with `to`
  (`try_to` reports running out of memory), are created, copied, filled, converted between
  dtypes (with exactly the CPU's results) and read back there. Elementwise operations, the
  optimizers' steps and random sampling run on the GPU with the CPU's results (bit for bit, apart
  from the last place of transcendental functions), and so do reductions and softmax
  (accumulated in f64 like the CPU's, in another order, with the GPU's exp and log in softmax);
  matrix products, indexing, pooling and convolutions don't yet.

Not available yet: most operations on GPUs (ROCm comes first), most neural network layers,
autograd hooks, and in-place operations on views that require grad.

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

## Building with ROCm

The ROCm backend is behind the `rocm` feature, off by default:

```toml
[dependencies]
homura = { version = "0.4", features = ["rocm"] }
# or, with the models: homura-models = { version = "0.4", features = ["rocm"] }
```

Building it needs ROCm with `hipcc` (it is developed with ROCm 10.1). The build looks for ROCm at
`ROCM_PATH`, then `hipconfig --rocmpath`, `/opt/rocm` and `/usr`. The supported GPU is gfx1201
(Radeon AI PRO R9700 and RX 9070 series); `HOMURA_OFFLOAD_ARCH` selects the architectures to
compile for, `gfx1201` by default. `gfx1200` is accepted too, but has only been compiled for, not
run: using such a GPU prints a warning. `HOMURA_ROCM_CHECK_ONLY=1` skips the native build, for
`cargo check`, `clippy` and documentation on machines without ROCm (nothing can be linked then).

Programs that use homura with ROCm load the HIP runtime (`libamdhip64.so`) when they start, so
the ROCm library directory must be on the dynamic linker's path, for example with
`/etc/ld.so.conf.d/rocm.conf` (as AMD's installation guide sets up) or `LD_LIBRARY_PATH`.
Otherwise the program fails to start with "error while loading shared libraries:
libamdhip64.so.7". The tests and examples of homura and homura-models find the library without
it; build scripts of other crates that depend on homura can read the directory from
`DEP_HOMURA_ROCM_LIB`.

## Minimum supported Rust version

Rust 1.88 or later. Raising it is a minor version change.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
