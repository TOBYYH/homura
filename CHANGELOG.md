# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as Cargo applies it to 0.x versions.

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

[0.1.1]: https://github.com/TOBYYH/homura/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/TOBYYH/homura/releases/tag/v0.1.0
