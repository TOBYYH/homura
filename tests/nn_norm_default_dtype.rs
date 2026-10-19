//! Review round 57 (R67): the parameters that the `LayerNorm` builders add keep the options of
//! the layer's parameters when the default dtype has changed since the layer was made. The
//! default dtype is global state, so this test has its own binary.

use homura::DType;
use homura::nn::{LayerNorm, RmsNorm};

#[test]
fn added_parameters_keep_the_layer_dtype() {
    let norm = LayerNorm::new([3]).bias(false);
    let only_bias = {
        let mut n = LayerNorm::new([3]);
        n.weight = None;
        n
    };
    let plain = LayerNorm::new([3]).elementwise_affine(false);
    homura::set_default_dtype(DType::F64);
    let norm = norm.bias(true);
    assert_eq!(norm.bias.as_ref().unwrap().dtype(), DType::F32);
    let only_bias = only_bias.elementwise_affine(true);
    assert_eq!(only_bias.weight.as_ref().unwrap().dtype(), DType::F32);
    // Without a parameter to follow, the new ones have the default dtype, as documented.
    let plain = plain.elementwise_affine(true);
    assert_eq!(plain.weight.as_ref().unwrap().dtype(), DType::F64);
    assert_eq!(plain.bias.as_ref().unwrap().dtype(), DType::F64);
    let rms = RmsNorm::new([3])
        .elementwise_affine(false)
        .elementwise_affine(true);
    assert_eq!(rms.weight.as_ref().unwrap().dtype(), DType::F64);
    homura::set_default_dtype(DType::F32);
}
