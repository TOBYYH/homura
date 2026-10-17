# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as Cargo applies it to 0.x versions.

## [Unreleased]

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

[Unreleased]: https://github.com/TOBYYH/homura/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/TOBYYH/homura/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/TOBYYH/homura/releases/tag/v0.1.0
