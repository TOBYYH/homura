# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as Cargo applies it to 0.x versions.

## [Unreleased]

### Added

- Shape, dimension and index arguments take `i32` and `usize` as well as `i64`, through the new
  sealed `DimSize` trait: `IntoShapeSpec` (`view`, `reshape`, `expand`) and `Dims` (`permute`,
  `sum_dim`, `squeeze_dim`, `flip`, …) accept `[T; N]`, `&[T; N]`, `&[T]` and `Vec<T>` of each,
  `Dims` and `IndexItem` (`index`) accept single values of each, and `IndexItem` accepts every
  range and `step_by` range of each, except open stepped ranges (`(a..).step_by(n)`), which stay
  `i64` only: advancing an open range of a narrower type past its largest value overflows inside
  the standard library. Sizes from `size()` or `shape()` need no `as i64`:
  `x.view([batch * seq, hidden])`, `x.index((.., ..n))`. `IntoShapeSpec` also accepts tuples of
  up to 8 sizes whose types may differ, as in `x.view((batch, seq, -1))`, and `()` for a
  zero-dimensional shape. A `usize` above `i64::MAX` panics.
- `optim::Optimizer` is object safe: a training loop, and later learning rate schedules and
  gradient clipping, can drive any optimizer through `&mut dyn Optimizer` or `Box<dyn Optimizer>`,
  also several of different types in one `Vec`. It has new methods for the groups:
  `num_param_groups`, `params(group)`, `lr(group)` and `set_lr(group, lr)`, which changes the
  learning rate of that group only, after checking the group's options with it like
  `add_param_group` (the group is unchanged if they are invalid). `Box<O>` and `&mut O` implement
  `Optimizer` and `TypedOptimizer` for an `O` that does, trait objects included.
- `autograd::grad_array`: `autograd::grad` for a fixed number of inputs, which returns an array
  to destructure: `let [dx, dw] = autograd::grad_array(&[&loss], [&x, &w], options)`. The
  inputs are an array literal of `&Tensor`, so `&Parameter`s dereference as with `grad`, whose
  signature is unchanged. `Function::apply_array` does the same for the outputs of a custom
  function, `let [y] = MyFunction.apply_array(&[&x])`, and panics if `forward` returns another
  number of outputs (checked after the call, which is then recorded).
- Closure forms of the scopes: `with_no_grad(|| ..)`, `with_enable_grad`, `with_grad_enabled`
  and `with_device(device, || ..)` run a closure with that grad mode or default device and
  return its value; the previous state is restored also when the closure panics. Unlike the
  guards of `no_grad` and `device_scope`, they can't be ended early by `let _ = ..`.
- `Tensor::index_put_accumulate` and `index_put_accumulate_` (PyTorch's `accumulate=True`), now
  that `index_put` / `index_put_` have no `accumulate` argument.
- `Module::load_state_dict_with(&sd, options)` with `nn::LoadOptions`, which is built from its
  default with setters, `LoadOptions::default().strict(false)` (PyTorch's `strict=False`), and
  read with getters such as `is_strict()`. Its fields are private, so that later options (such
  as `assign`) can be added without breaking callers.
- `ValuesAndIndices { values, indices }`, which `max_dim` and `min_dim` now return (PyTorch's
  `torch.return_types.max`); `into()` converts it to the former tuple.
- `Sum` and `Product` for `Tensor`, over tensors and references to them:
  `losses.iter().sum::<Tensor>()` is Python's `sum(losses)`. The start is a plain integer 0 or 1,
  so the dtype and device of the result come from the tensors (an `I32` sum stays `I32`; one
  tensor gives a new tensor recorded as `AddBackward0` / `MulBackward0`). Without tensors, the
  result is a zero-dimensional `I64` tensor on the default device.
- `Module::display()`, PyTorch's `print(model)`: `println!("{}", model.display())` prints the
  module tree in PyTorch's format, with each module's type name and settings, a submodule shared
  through an `Arc` under each of its names, and runs of equal elements of a `Vec` or an array
  grouped as by `nn.ModuleList` (`(0-2): 3 x Linear(...)`). The built-in layers print PyTorch's
  settings (`Linear(in_features=4, out_features=8, bias=True)`) under homura's type names
  (`Relu`, `RmsNorm`). `nn::ModuleDisplay::new(&dyn Module)` prints the modules that
  `named_modules` lists.
- `Module::extra_repr` and `Module::type_name`, which `display` prints, and the struct attribute
  `#[module(extra_repr = Self::describe)]` of `#[derive(Module)]`, which implements
  `extra_repr` with the given function. `Box`, `Arc` and `Option` give the name and settings of
  the module they hold; `Vec` and arrays are named `ModuleList`, `BTreeMap` `ModuleDict`.
- `Module::num_parameters()`: the number of elements of the parameters, each parameter once
  (`sum(p.numel() for p in model.parameters())`).

  The new provided methods of `Module` (`display`, `extra_repr`, `type_name`,
  `num_parameters`, and `version` and `added_in` below) can make a call of a method of the same name from another trait in scope
  ambiguous; inherent methods of the same name keep precedence.
- Loading by assignment, PyTorch's `load_state_dict(sd, assign=True)`:
  `LoadOptions::default().assign(true)` (read with `is_assign()`) gives the parameters and
  buffers the dtype, device, layout and storage of the entries instead of copying, so a model
  built on the meta device (`with_device(Device::Meta, ..)`) loads without allocating twice.
  Unlike PyTorch, the parameters and buffers stay the same objects: optimizers and shared
  names see the new data, a parameter keeps whether it requires grad and loses its `.grad`, and
  graphs and views of the old data panic instead of using it. Targets that are views or have a
  history, and integer entries for parameters that require grad, are load errors.
- Module versions, PyTorch's `_version` and `state_dict._metadata`: `Module::version()` (1 by
  default, set with the struct attribute `#[module(version = N)]`), recorded by `state_dict()`
  for every module path in the new `nn::ModuleVersions`, read and changed with
  `StateDict::versions`, `versions_mut`, `module_version` and `set_module_version`. State dicts
  collected from entries (from safetensors, for example) have none.
- Old state dicts without a buffer that a module added in a later version: the field attribute
  `#[module(added_in = N)]` (`Module::added_in`), for zero-dimensional integer buffers. When such
  a key is missing and the state dict's version of the module is lower or absent, loading keeps
  the buffer (on the meta device, assignment gives it a 0 on the CPU). `BatchNorm2d` is version 2
  with `num_batches_tracked` added in 2, so it loads PyTorch state dicts saved before the count
  existed and plain state dicts without it, as PyTorch does; a state dict of version 2 without
  the count still reports it missing.
- Key transforms on `StateDict`: `consume_prefix` (PyTorch's
  `consume_prefix_in_state_dict_if_present`, for `module.*` keys), `add_prefix`, `sub(prefix)`,
  the state dict of a submodule, and `map_keys(f)`, which renames or drops each key (and drops
  the versions). Versions follow the prefixes.
- `homura::load(path)` and `homura::save(&state_dict, path)` for model files. `load` tells the
  format from the content: a safetensors file, a PyTorch `.pt` file (see below), or a sharded
  checkpoint's JSON index (`model.safetensors.index.json` or `pytorch_model.bin.index.json`),
  whose shards it reads in `weight_map` order after checking that the index and the shards agree
  and that shard names stay in the index's directory. `save` picks the format by the extension
  (`.safetensors`; `.pt` files can't be saved yet).
- The module `homura::io` is public, with `io::save_sharded(&sd, dir, ShardOptions)`, which
  splits a state dict into shards and an index in the Hugging Face layout (or one file if it
  fits) and replaces an earlier checkpoint with the same prefix in that directory: it removes
  the entry point of the other layout, which loaders could pick instead of the new files, and
  the old shards that the earlier index lists. `ShardOptions::prefix` refuses prefixes that end
  like a numbered shard (`model-00001-of-00002`), so different prefixes never share file names;
  `io::safetensors_metadata(path)` and `io::save_safetensors_with_metadata(&sd, path,
  &metadata)` for `__metadata__`, and the four existing safetensors functions, which keep their
  signatures and stay at the crate root.
- safetensors files are read from the file straight into each tensor's storage instead of being
  read into memory as a whole first, so loading needs about the size of the tensors, not twice
  that. Saving writes one tensor at a time.
- `unsafe fn io::load_mmap(path)`, like `torch.load(path, mmap=True)`: loads a safetensors file or
  a sharded checkpoint's index as `load` does, but the tensors use a private copy-on-write
  mapping of the files instead of copies (new dependency `memmap2`). Writes to the tensors stay
  in the process; a file is unmapped when the last tensor using it is dropped. It is `unsafe`
  because the files must not change while any tensor using them is alive; `load` and the other
  safe functions never map files. Tensors whose data isn't aligned to their element size in the
  file or is in another byte order than the host's are copied, and BOOL data is checked as for
  `load` (safetensors) or copied (`.pt`).
- Reading PyTorch's `.pt` format, as `torch.load(path, map_location="cpu", weights_only=True)`
  does: `load` and `load_mmap` read state dicts, including `pytorch_model.bin`, and the new
  `io::load_value(path)` and `io::load_value_from_bytes(bytes)` read any checkpoint of the
  supported objects (for example `{"model": .., "optimizer": .., "epoch": 3}`) as an `io::Value`.
  Tensors keep the file's views: tensors that share a storage in the file share it after
  loading (with its version counter), with the file's offsets and strides; a tensor referenced
  several times is the same handle; each storage is read once, straight into its memory, and
  swapped if the file's byte order isn't the host's. Tensors keep the `requires_grad` saved with
  them, a `Parameter` becomes a plain tensor, and BOOL bytes other than 0 and 1, which PyTorch
  writes and reads as `True`, are read as 1. Everything is checked before a tensor is made:
  storage records must exist and have the declared size, a storage declared twice must be
  declared alike, views must lie inside their storage, and only floating point tensors can
  require grad. The zip archive is checked too (stored entries, ZIP64, the data descriptors that
  `torch.save` writes, CRCs unless they are 0, which means the writer didn't compute them), and
  its layout (`version` 1 to 10, `byteorder`). The pickle is read with a restricted unpickler
  that runs no code: it accepts only `collections.OrderedDict`, PyTorch's tensor rebuild
  functions and storage classes, pickle protocols 2 to 5, and int and str dict keys, and limits
  how far objects referenced several times can expand, so a small file can't make it allocate
  without bound; cycles are an error. TorchScript archives and the format before PyTorch 1.6
  get errors that say so.
- `io::Value`, `io::Dict` and `io::Key`: the Python objects of checkpoints (None, bool, int,
  float, str, list, tuple, dict or `OrderedDict` with int or str keys, and tensors). An
  `OrderedDict` keeps the module versions of a state dict (PyTorch's `_metadata`), also inside a
  checkpoint: `Value::from(&state_dict)` and `StateDict::try_from(&value)` convert with them,
  and a plain dict gives a state dict without versions.

### Changed

- Graphs recorded before `Module::to` stay refused after converting back to the original dtype
  and device: a backward pass that reaches a converted parameter's old accumulator panics, as
  one that unpacked it already did. It used to run a graph that didn't save the parameter.
  The messages also cover loading by assignment.
- **Breaking**: the methods of `optim::Optimizer` that use the optimizer's options type moved to
  the new trait `optim::TypedOptimizer: Optimizer`, which has the associated type `Options`:
  `defaults`, `param_groups`, `options_mut` and `add_param_group`. Code that calls them needs
  `use homura::optim::TypedOptimizer`, and `dyn Optimizer<Options = SgdOptions>` becomes
  `dyn TypedOptimizer<Options = SgdOptions>`, which converts to `dyn Optimizer`.
- **Breaking**: since shape, dimension and index arguments have several integer types to choose
  from, an unsuffixed integer literal in them now has the type `i32`, Rust's fallback, instead of
  `i64`, and a list without literals has no type to infer. Three kinds of code no longer compile:
  - a literal beyond the `i32` range, such as `x.index(..5_000_000_000)` or
    `x.expand([1 << 40, 4])`: add a suffix (`5_000_000_000i64`, `1i64 << 40`);
  - a variable whose type was inferred as `i64` from such an argument and is then used in a way
    that needs a known type, such as `let i = 3; x.index(i); i.pow(2)`: annotate it
    (`let i: i64 = 3`);
  - an empty list: `view([])`, `reshape(&[])`, `expand(Vec::new())`, `sum_dim([], false)`,
    `permute([])`: write `view(())` or `view(Shape::scalar())` for a zero-dimensional shape and
    `[] as [i64; 0]` for an empty list of dimensions. `zeros([])` and the other functions that
    take an `IntoShape` are not affected.
- **Breaking**: `Tensor::index_put(indices, values)` and `index_put_` lost their `accumulate`
  argument; `accumulate=True` is `index_put_accumulate` / `index_put_accumulate_`. Replace
  `x.index_put_(i, &v, false)` by `x.index_put_(i, &v)` and `x.index_put_(i, &v, true)` by
  `x.index_put_accumulate_(i, &v)`. The autograd node stays `IndexPutBackward0`.
- **Breaking**: `Module::load_state_dict(&sd)` is always strict (PyTorch's default) and lost its
  `strict` argument. Replace `load_state_dict(&sd, true)` by `load_state_dict(&sd)` and
  `load_state_dict(&sd, false)` by
  `load_state_dict_with(&sd, LoadOptions::default().strict(false))`. Both return
  `Result<LoadReport, LoadError>` as before.
- **Breaking**: `Tensor::max_dim` and `min_dim` return `ValuesAndIndices` instead of a tuple.
  Replace `.0` / `.1` by `.values` / `.indices`, and `let (v, i) = x.max_dim(..)` by
  `let ValuesAndIndices { values, indices } = x.max_dim(..)` or
  `let (v, i) = x.max_dim(..).into()`. The values, indices and gradients are unchanged.
  `max_pool2d_with_indices` still returns a tuple, as in PyTorch.

- A panic for an invalid argument (shape, dtype, device, dimension, index or option) now reports
  the location of the caller's code instead of a line inside homura: the public API carries
  `#[track_caller]`, including tensor methods and operators, the crate's free functions,
  `nn::functional`, layers and their `Forward` implementations, `Module` methods, optimizers
  (also through `dyn Optimizer`), `autograd::backward` / `grad`, `Tensor::backward`,
  `Function::apply` and `gradcheck`, and the internal checks they share, such as index and dtype
  checks in the CPU kernels and the rejection of a view whose base was modified in place after
  the view was created. Errors found while the backward pass runs (a saved tensor modified in
  place, a graph run twice) are still reported inside homura. Methods that homura calls on user
  implementations (`Function::forward` / `backward`, `Visitor`, `Member::visit_as`,
  `Module::visit`) don't carry the attribute, so panics in user code keep their own location.

## [0.3.0] - 2026-10-22

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
- `nn::functional::pad` in PyTorch's constant mode, with negative pads that crop, recorded as
  `ConstantPadNdBackward0` (its gradient pads again). The fill value is converted and checked
  like `fill_`, only when something is filled. Unlike PyTorch, the result is always contiguous.
- `Tensor::unfold`, a view of the windows along a dimension, recorded as `UnfoldBackward0`. Its
  gradient adds overlapping windows in the dtype in PyTorch's order and is recorded as
  `UnfoldBackwardBackward0`. Unlike PyTorch, in-place writes into overlapping windows panic.
- `nn::functional::conv2d` with `Conv2dOptions` (stride, `Conv2dPadding::Explicit` / `Valid` /
  `Same`, dilation, groups), for batched and unbatched inputs, floating point and integer
  dtypes, computed from padding, `unfold` and a matrix product, so it is differentiable to any
  order; half precision is computed in `F32` and rounded once, as in PyTorch. Unlike PyTorch,
  the graph holds those operations rather than one `ConvolutionBackward0`, and inputs without
  channels panic.
- `nn::functional::max_pool2d` and `max_pool2d_with_indices` with `MaxPool2dOptions` (kernel
  size, stride, padding, dilation, `ceil_mode`), with PyTorch's selection rules (the first of
  equal maxima, the last NaN, never the padding) and checks, recorded as
  `MaxPool2DWithIndicesBackward0`, whose gradient adds in PyTorch's order and is differentiable
  again. Unlike PyTorch, windows that cover only padding panic.
- `nn::functional::adaptive_avg_pool2d`, a mean over the last two dimensions for an output of
  `[1, 1]` and otherwise PyTorch's kernel, recorded as `AdaptiveAvgPool2DBackward0`, with
  PyTorch's order of operations in both directions.
- `nn::functional::batch_norm` with `BatchNormOptions` (`training`, `momentum`, `eps`), with or
  without running statistics, weight and bias, for inputs of 2 or more dimensions; half
  precision inputs may have `F32` parameters and statistics. It is a single
  `NativeBatchNormBackward0` node with PyTorch's CPU formulas, differentiable again. Unlike
  PyTorch, updating the running statistics increments their versions, so an evaluation graph
  that saved them can't run backward after an update, and statistics to update must be writable
  and share no memory with the input, the parameters or each other; parameters and statistics
  need only `C` elements, also in the backward pass; inputs without elements are checked like
  others.
- The layers `nn::Conv2d` (stride, padding, dilation, groups, bias), initialized like PyTorch's;
  `nn::BatchNorm2d` with PyTorch's buffers `running_mean`, `running_var` and
  `num_batches_tracked`, momentum or a cumulative average (`momentum(None)`), and the options
  `affine`, `bias` and `track_running_stats`; `nn::MaxPool2d` (with `forward_with_indices`),
  `nn::AdaptiveAvgPool2d` and `nn::Flatten`. Their options are set one after another
  (`Conv2d::new(3, 8, 3).padding(1)`), with sizes as a number or a pair (`nn::IntoPair`), and
  checked like PyTorch's constructors when they are set. Unlike PyTorch, a `BatchNorm2d` state
  dict without `num_batches_tracked` is missing a key.

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

[Unreleased]: https://github.com/TOBYYH/homura/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/TOBYYH/homura/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/TOBYYH/homura/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/TOBYYH/homura/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/TOBYYH/homura/releases/tag/v0.1.0
