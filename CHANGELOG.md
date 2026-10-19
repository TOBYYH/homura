# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as Cargo applies it to 0.x versions.

## [Unreleased]

### Added

- The `nn` module's core: `Parameter`; the `Module` and `Member` traits, with `Visitor` and
  `SharedId`; PyTorch's traversals (`named_parameters`, `named_buffers`,
  `named_modules`, `named_children` and the unnamed versions), `train` / `eval` through
  `Training` flags, `zero_grad` and `requires_grad_`; `StateDict`, `state_dict` and
  `load_state_dict`, which checks every entry before it writes any; and `Module::to`, which
  converts parameters and buffers in place, keeping them the same tensors.
- `#[derive(Module)]` (`nn::Module`, from the new `homura-derive` crate) for structs with named
  fields or none, generic ones included, with the field attributes `rename`, `skip` and
  `non_persistent` and the struct attribute `crate`.
- `nn::Sequential` and the `nn::Forward` trait of its layers.
- `nn::init` with PyTorch's initializations (`uniform_`, `normal_`, `constant_`, `ones_`,
  `zeros_`, `kaiming_uniform_`, `kaiming_normal_`, `xavier_uniform_`, `xavier_normal_`) and
  `calculate_gain` / `calculate_fan_in_and_fan_out`.
- `nn::functional` with `linear`, which records the operations of PyTorch's `F.linear`, and the
  activations `relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `softmax` and `log_softmax`.
- The layers `nn::Linear`, initialized like PyTorch's, and `nn::Relu`, `nn::Gelu`, `nn::Silu`,
  `nn::Tanh`, `nn::Sigmoid` and `nn::Softmax`.
- The losses `nn::functional::mse_loss`, `nll_loss`, `cross_entropy` (class and probability
  targets, `weight`, `ignore_index`, `label_smoothing`, through `CrossEntropyOptions`) and
  `binary_cross_entropy_with_logits` (`weight`, `pos_weight`), with a `Reduction`, recording
  PyTorch's backward nodes and gradient formulas.
- `Tensor::log_sigmoid` and `nn::functional::logsigmoid`, computed stably like PyTorch's kernel.
- `nn::functional::embedding` (`padding_idx`) and the `nn::Embedding` layer, a single
  `EmbeddingBackward0` node whose padding row gets no gradient, as in PyTorch.
- `nn::functional::layer_norm` and the `nn::LayerNorm` layer, a single
  `NativeLayerNormBackward0` node with PyTorch's CPU formulas (half precision inputs may have
  `F32` parameters); `nn::functional::rms_norm` and the `nn::RmsNorm` layer, which record the
  operations of PyTorch's CPU implementation (the default `eps` is the machine epsilon).
- `nn::functional::dropout` and the `nn::Dropout` layer (with a `Training` flag), computed like
  PyTorch's CPU implementation; the masks come from homura's random number generator.
- The `optim` module: the `Optimizer` trait with `ParamGroup`s, and `Sgd` (momentum, dampening,
  weight decay, Nesterov, maximize), `Adam` and `AdamW` (betas, eps, weight decay, AMSGrad,
  maximize), with PyTorch's defaults, checks and update order. Each compound operation of a step
  is computed like PyTorch's vectorized CPU kernels and rounded once, which matched the reference
  build bit for bit in the tested cases, except that F16 / BF16 scaled additions past the last
  multiple of 32 elements don't round after each operation as PyTorch's scalar loop does. Scalars
  that overflow their type are rejected as there. Only the options of a group can be changed
  afterwards (`options_mut`). Unlike PyTorch, a parameter listed twice in a group is an error, and
  `step` checks the options, the parameters and their state before it updates anything. Optimizer
  state can't be saved or loaded yet.
- Advanced indexing with `Tensor::index`: any number of `i64` / `i32` index tensors and `bool`
  masks, mixed with the other items and broadcast together, with PyTorch's rules for where the
  indexed dimensions go; `bool` items (`x.index(true)`). The backward graph holds a single
  `IndexBackward0`, as in PyTorch. Index tensors may be on the CPU for a tensor on any device.
- `Tensor::index_put` and `Tensor::index_put_` (with `accumulate`), taking index tensors, full
  slices `..` and an `Ellipsis`, recorded as `IndexPutBackward0`. Unlike PyTorch, the in-place
  version checks every position before it writes anything, and rejects a target whose elements
  may share memory instead of warning.
- Custom differentiable functions: the `autograd::Function` trait (`forward`, `backward` and the
  provided `apply`) with `ForwardContext` (`save_for_backward`, `mark_non_differentiable`,
  `needs_input_grad`) and `BackwardContext` (`saved_tensors`, `needs_input_grad`), recorded as
  one node named after the type (`MyReluBackward`), like PyTorch's `torch.autograd.Function`.
  Saved inputs and outputs are checked, freed and differentiable again like the tensors of
  built-in operations; other saved tensors keep their history, if any. Intermediates computed
  with grad mode off are constants. Unlike PyTorch, outputs that share
  memory with an input or another output, in-place modifications of the inputs and marking a
  tensor that isn't an output panic; outputs without a gradient give `backward` `None` instead of
  zeros, and `backward` isn't called when no output has one.

### Changed

- A backward pass sums a gradient of a larger shape that its input's shape broadcasts to, as
  PyTorch's engine does, and panics like PyTorch on a gradient on another device unless it is
  zero-dimensional (which is moved there), also for the gradients given to `backward` and
  `autograd::grad`; it used to panic on any other shape and move any gradient. Built-in
  operations always return gradients of the input's shape and device.

- `Tensor::index` with an integer tensor no longer records an `UnsafeViewBackward0` on top of
  `IndexBackward0`. A zero-dimensional integer index tensor now selects, like PyTorch: the result
  is a view, recorded as `SelectBackward0`. Index tensors of dtypes other than `I64`, `I32` and
  `Bool` are rejected (`U8`, `I8` and `I16` used to index as integers; PyTorch takes `U8` tensors
  as deprecated masks and zero-dimensional `I8` / `I16` tensors as positions). Out-of-range
  messages name the position of the index tensor among the indexed dimensions, as PyTorch's do.

- Subtraction (`sub`, `sub_`, the `-` operator) panics like PyTorch when either operand is a bool
  tensor or scalar; it used to promote a single bool operand (`[2, 3] - [true, false]` gave
  `[1, 3]`). Meta tensors are checked too.
- `fill_`, `full` (with `full_with`, `new_full` and `full_like`), `masked_fill` and
  `masked_fill_` with a scalar panic like PyTorch when the value overflows the dtype (`300` for
  `U8`, `1e40` for `F32`), also for tensors without elements and for a zero-dimensional tensor
  `value` of `masked_fill`; they used to write a wrapped or infinite value. Meta tensors take
  any value. Unlike PyTorch, a single `F16` or `BF16` element is checked too.
- A backward pass panics instead of writing a gradient that doesn't match its leaf, when
  `Module::to` converted the leaf (or the tensor that is its `.grad`) after the graph was
  recorded; views whose base was converted are stale.

## [0.2.0] - 2026-10-17

### Added

- Automatic differentiation in the now public `autograd` module, following PyTorch's autograd:
  operations on tensors that require grad record a backward graph, which `autograd::backward`
  and `autograd::grad` (with `BackwardOptions` and `GradOptions`) run. `retain_graph` keeps the
  saved tensors for another pass, and `create_graph` records the pass itself, for gradients of
  gradients to any order. Backward passes on different threads can share leaves.
- `Tensor` methods `requires_grad_`, `is_leaf`, `grad`, `set_grad`, `grad_fn`, `retain_grad`,
  `retains_grad`, `detach`, `detach_`, `backward` and `backward_with`, and the `GradFn` handle
  (`name`, `next_functions`, `is_same`).
- Gradient formulas for every differentiable operation, with PyTorch's node names and saved
  tensors, following PyTorch's rules at NaN, infinities, ties and other boundaries. Floating
  point CPU reductions and inner products accumulate in f64, so extreme cancellation can produce
  different finite / infinity / NaN results. The gradient of `prod` for an input with zeros
  can't be differentiated again yet.
- In grad mode, the 17 in-place operations are recorded like their out-of-place versions, and
  backward passes check with version counters that the tensors they saved weren't modified, with
  PyTorch's errors. Recorded in-place operations on views, and views whose base was modified in
  place after they were created, are rejected with an error instead (not supported yet).
- `autograd::gradcheck` and `autograd::gradgradcheck`, with `GradcheckOptions` and
  `GradcheckError`, compare gradients with finite differences.
- New operations: `erf`, `sign`, `index_add`, `index_add_`, `gather`, `scatter_add`,
  `scatter_add_`, `select_scatter`, `slice_scatter` and `sum_to_size`.
- `Debug` output ends with `grad_fn=<Name>` or `requires_grad=True`, like PyTorch's `repr`.
- The minimum supported Rust version, 1.88, is declared in `rust-version` and checked in CI.

### Changed

- **Breaking:** `squeeze_dim` with an empty list of dimensions squeezes nothing, as in PyTorch;
  it squeezed every dimension of size 1.
- **Breaking:** an index expression that selects the whole tensor, such as `(.., ..)` or
  `Ellipsis`, returns a new view of it instead of another handle to the same tensor.
- **Breaking:** `squeeze_dim`, `sum_dim`, `mean_dim`, `var_dim`, `std_dim`, `all_dim`, `any_dim`
  and `flip` reject repeated dimensions of zero-dimensional tensors, such as `[0, -1]`, as
  PyTorch does.
- **Breaking:** `alpha` and `beta` of any scalar kind are converted to the type PyTorch's kernel
  takes them in, and finite values out of its range are rejected: f32 for f16, bf16 and f32
  inputs and f64 for f64 inputs in `addmm`, `baddbmm` and `addbmm` (so 1e39 is rejected for
  f32 inputs), the input dtype in `addmv` and `addr` (so an f16 `addmv` rejects 65505 and
  takes the integer 2049 as 2048).
- `requires_grad` reports whether gradients flow to the tensor instead of always `false`.
- `mul` and `div` of f16 and bf16 tensors by a scalar compute in f32 and round once, as
  PyTorch's CPU kernels do, so results can differ from earlier ones by a rounding step.
- `Debug` output wraps the line before its suffixes where PyTorch's `repr` does.

### Fixed

- For f32 and f64 `addmm`, `addmv`, `baddbmm` and `addbmm`, a zero `alpha` gives
  `beta * input` even where the product overflows. Their f16 and bf16 matrix products are
  kept in f32 until they have been scaled, then rounded to the output dtype (CPU inner
  products still accumulate in f64). `addr` multiplies `alpha` by `vec1` before `vec2`, so
  a tiny `alpha` no longer turns a finite result infinite. These rules are independent of
  size and layout; some PyTorch CPU kernel paths differ.
- `uniform_` and `normal_` check that the target can be written before drawing samples, so a
  rejected call doesn't advance the random number generator.
- Inclusive index ranges ending at `i64::MAX` reach the end of the dimension instead of
  panicking in debug builds and selecting nothing in release builds.
- `split` with a size of 0 and `chunk` of an empty dimension return a separate view for every
  piece instead of the input itself or one shared view.
- Empty tensors: `matmul` of `[0, 0]` and `[0, n]` could fail with a capacity overflow, and
  reductions, `softmax`, `log_softmax`, `triu`, `tril`, `matmul`, `index_select` and integer
  tensor indexing could overflow computing offsets from the strides of empty views.

## [0.1.1] - 2026-10-11

### Fixed

- `uniform_` requires both bounds and their difference to be finite values of the destination
  dtype, as PyTorch does.
- `randint` with a floating point dtype rejects intervals outside the range of exactly
  representable integers (±2^11 for f16, ±2^8 for bf16, ±2^24 for f32, ±2^53 for f64), where it
  could previously return values outside `[low, high)`. Rejected calls don't advance the random
  number generator.
- `addmm`, `addmv`, `baddbmm` and `addbmm` convert `beta` and `alpha` to the input dtype instead
  of promoting the result: integer inputs now give integer results, as in PyTorch, with floating
  point coefficients truncated toward zero and coefficients that overflow the dtype rejected.
- `addr` promotes all three inputs to a common dtype before computing, supports bool inputs, and
  rejects floating point coefficients for integer or bool results, as PyTorch does.
- The strided CPU iterator no longer overflows when constructed for empty views.
- JSON `\u` escapes (in safetensors headers) require exactly four hexadecimal digits.
- The documentation of float-to-integer casts states exactly where results match PyTorch.

## [0.1.0] - 2026-10-11

### Added

- First release: `Tensor` with reference semantics, dtypes and type promotion, views, indexing,
  broadcasting, forward operators on the CPU reference backend, the meta device, random number
  generation and minimal safetensors reading and writing.

[Unreleased]: https://github.com/TOBYYH/homura/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/TOBYYH/homura/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/TOBYYH/homura/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/TOBYYH/homura/releases/tag/v0.1.0
