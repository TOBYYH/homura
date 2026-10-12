//! `matmul` on operands of every rank and layout: vectors, broadcast batch dimensions that can't be
//! merged without copying, transposed, sliced and expanded operands, and empty dimensions, on the
//! CPU and the meta device.

use homura::{DType, Device, Tensor};

/// `[0, 1, ..., n)` with `shape` on `device`.
fn operand(shape: &[usize], device: Device) -> Tensor {
    let n: usize = shape.iter().product();
    let shape: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    homura::arange_with(n as i64, device).view(shape)
}

/// `a @ b` computed by broadcasting and summing instead of `matmul`.
fn reference(a: &Tensor, b: &Tensor) -> Tensor {
    let a2 = if a.num_dims() == 1 {
        a.unsqueeze(0)
    } else {
        a.clone()
    };
    let b2 = if b.num_dims() == 1 {
        b.unsqueeze(-1)
    } else {
        b.clone()
    };
    // [..., m, k, 1] * [..., 1, k, n], summed over k.
    let mut c = (a2.unsqueeze(-1) * b2.unsqueeze(-3)).sum_dim(-2, false);
    if a.num_dims() == 1 {
        c = c.squeeze_dim(-2);
    }
    if b.num_dims() == 1 {
        c = c.squeeze_dim(-1);
    }
    c
}

fn values(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

/// Checks `a @ b` against [`reference`] on the CPU, then its shape and dtype on the meta device.
/// The operands are built on each device by `a` and `b`, so both see the same layouts.
fn check(a: impl Fn(Device) -> Tensor, b: impl Fn(Device) -> Tensor, shape: &[usize]) {
    let (x, y) = (a(Device::Cpu), b(Device::Cpu));
    let c = x.matmul(&y);
    assert_eq!(c.shape(), shape);
    assert_eq!(c.dtype(), x.dtype());
    assert!(c.is_contiguous());
    let expected = reference(&x, &y);
    assert_eq!(expected.shape(), shape);
    assert_eq!(values(&c), values(&expected));

    let (x, y) = (a(Device::Meta), b(Device::Meta));
    let c = x.matmul(&y);
    assert_eq!(c.shape(), shape);
    assert_eq!(c.dtype(), x.dtype());
    assert_eq!(c.device(), Device::Meta);
}

/// Checks `a @ b` for operands built by [`operand`].
fn check_shapes(a: &[usize], b: &[usize], shape: &[usize]) {
    check(|d| operand(a, d), |d| operand(b, d), shape);
}

#[test]
fn vector_operands() {
    check_shapes(&[4], &[4], &[]);
    check_shapes(&[4], &[4, 3], &[3]);
    check_shapes(&[4], &[2, 4, 3], &[2, 3]);
    check_shapes(&[3, 4], &[4], &[3]);
    check_shapes(&[2, 3, 4], &[4], &[2, 3]);
    check_shapes(&[2, 1, 3, 4], &[4], &[2, 1, 3]);
}

#[test]
fn broadcast_batch_dimensions() {
    // The left operand is expanded to the batch [2, 5] with a zero stride inside it, so the
    // batch can't be merged into one dimension without copying.
    check_shapes(&[2, 1, 3, 4], &[1, 5, 4, 6], &[2, 5, 3, 6]);
    check_shapes(&[3, 4], &[2, 2, 4, 5], &[2, 2, 3, 5]);
    check_shapes(&[2, 1, 3, 4], &[4, 5], &[2, 1, 3, 5]);
    check_shapes(&[5, 1, 3, 4], &[2, 4, 6], &[5, 2, 3, 6]);
    check_shapes(&[1, 3, 4], &[2, 4, 5], &[2, 3, 5]);
}

#[test]
fn transposed_sliced_and_expanded_operands() {
    check(
        |d| operand(&[2, 4, 3], d).transpose(1, 2),
        |d| operand(&[2, 5, 4], d).transpose(1, 2),
        &[2, 3, 5],
    );
    // Transposed batch dimension.
    check(
        |d| operand(&[3, 2, 4], d).transpose(0, 1),
        |d| operand(&[2, 4], d).transpose(0, 1),
        &[2, 3, 2],
    );
    // Stepped slice and narrowed columns.
    check(
        |d| operand(&[2, 3, 8], d).index((.., .., (0..8).step_by(2))),
        |d| operand(&[4, 6], d).narrow(1, 1, 3),
        &[2, 3, 3],
    );
    // Transposed and broadcast.
    check(
        |d| operand(&[4, 3], d).transpose(0, 1),
        |d| operand(&[2, 1, 6, 4], d).transpose(-1, -2),
        &[2, 1, 3, 6],
    );
    // Batch dimension expanded with a zero stride.
    check(
        |d| operand(&[1, 3, 4], d).expand([5, 3, 4]),
        |d| operand(&[5, 4, 2], d),
        &[5, 3, 2],
    );
    check(
        |d| operand(&[4], d).expand([3, 4]),
        |d| operand(&[4, 2], d).expand([2, 4, 2]),
        &[2, 3, 2],
    );
}

#[test]
fn empty_dimensions() {
    check_shapes(&[0, 3, 4], &[4, 5], &[0, 3, 5]);
    check_shapes(&[2, 3, 0], &[0, 5], &[2, 3, 5]);
    check_shapes(&[2, 0, 4], &[4, 5], &[2, 0, 5]);
    check_shapes(&[2, 3, 4], &[4, 0], &[2, 3, 0]);
    check_shapes(&[0, 1, 3, 4], &[1, 5, 4, 6], &[0, 5, 3, 6]);
    check_shapes(&[0], &[0], &[]);
    check_shapes(&[0], &[0, 5], &[5]);
    check_shapes(&[3, 0], &[0], &[3]);
}

#[test]
fn float_operands() {
    let f32_operand = |shape: &'static [usize]| {
        move |d: Device| {
            let n: usize = shape.iter().product();
            let shape: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
            homura::arange_with(n as i64, (DType::F32, d)).view(shape)
        }
    };
    check(
        f32_operand(&[2, 1, 3, 4]),
        f32_operand(&[1, 5, 4, 6]),
        &[2, 5, 3, 6],
    );
    check(
        |d| f32_operand(&[2, 4, 3])(d).transpose(1, 2),
        f32_operand(&[4]),
        &[2, 3],
    );
}

/// The attention scores of a Llama layer with grouped-query attention: keys repeated with
/// `unsqueeze`, `expand` and a copying `reshape`, or broadcast by `matmul` itself.
#[test]
fn grouped_query_attention() {
    let (b, s, heads, kv_heads, d) = (2, 3, 4, 2, 5);
    let n_rep = heads / kv_heads;
    let q = |dev| operand(&[b, s, heads, d], dev).transpose(1, 2);
    let k = |dev| operand(&[b, s, kv_heads, d], dev).transpose(1, 2);
    let repeated = |dev| {
        k(dev)
            .unsqueeze(2)
            .expand([b as i64, kv_heads as i64, n_rep as i64, s as i64, d as i64])
            .reshape([b as i64, heads as i64, s as i64, d as i64])
    };
    check(q, |dev| repeated(dev).transpose(2, 3), &[b, heads, s, s]);
    check(
        |dev| q(dev).view([b as i64, kv_heads as i64, n_rep as i64, s as i64, d as i64]),
        |dev| k(dev).unsqueeze(2).transpose(-1, -2),
        &[b, kv_heads, n_rep, s, s],
    );
    let scores = q(Device::Cpu).matmul(&repeated(Device::Cpu).transpose(2, 3));
    let grouped = q(Device::Cpu)
        .view([b as i64, kv_heads as i64, n_rep as i64, s as i64, d as i64])
        .matmul(&k(Device::Cpu).unsqueeze(2).transpose(-1, -2));
    assert_eq!(
        values(&scores),
        values(&grouped.reshape([b as i64, heads as i64, s as i64, s as i64]))
    );
    // Attention weights times the repeated values.
    check(
        |dev| operand(&[b, heads, s, s], dev),
        repeated,
        &[b, heads, s, d],
    );
}
