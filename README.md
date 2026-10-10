# homura

A tensor computing framework written in Rust that aims to feel as close to PyTorch as possible.
Each hardware backend is built directly on the vendor's latest official development tools, so that
it can get the best performance out of the hardware.

The first GPU backend targets AMD ROCm. Backends for other hardware will be added once the ROCm
backend is stable.

> This project is in early development: only the CPU reference backend and the meta device are
> implemented so far. The API is not stable yet and it is not ready for real-world use.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
