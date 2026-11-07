//! Gradients and modules across the CPU and a ROCm GPU (M4 step 2b): the backward of `to` and
//! `copy_` between devices, gradients that a custom function returns on another device, and
//! `Module::to` to the GPU and back. Needs a ROCm GPU; built only with the `rocm` feature.
#![cfg(feature = "rocm")]

use homura::autograd::{self, BackwardContext, ForwardContext, Function, GradOptions};
use homura::nn::{Module, Parameter};
use homura::{DType, Device, Tensor};

const GPU: Device = Device::Rocm(0);

#[test]
fn to_between_devices_sends_the_gradient_back() {
    // CPU leaf, GPU result with another dtype: the gradient comes back to the CPU as f32.
    let x = homura::ones([2, 3]);
    x.requires_grad_(true);
    let y = x.to((DType::F64, GPU));
    assert_eq!((y.device(), y.dtype()), (GPU, DType::F64));
    y.backward_with(&homura::full_with([2, 3], 2.0, (DType::F64, GPU)));
    let g = x.grad().unwrap();
    assert_eq!((g.device(), g.dtype()), (Device::Cpu, DType::F32));
    assert_eq!(g.to_vec::<f32>(), [2.0; 6]);

    // GPU leaf, CPU result: the gradient goes to the GPU.
    let xg = homura::ones_with([3], GPU);
    xg.requires_grad_(true);
    xg.to(Device::Cpu).backward_with(&homura::full([3], 3.0));
    let g = xg.grad().unwrap();
    assert_eq!(g.device(), GPU);
    assert_eq!(g.to_vec::<f32>(), [3.0; 3]);

    // Second order: the gradient of the gradient with respect to the vector of the product.
    let x = homura::ones([2]);
    x.requires_grad_(true);
    let y = x.to((DType::F64, GPU));
    let v = homura::full_with([2], 2.0, (DType::F64, GPU));
    v.requires_grad_(true);
    let g = autograd::grad(
        &[&y],
        &[&x],
        GradOptions {
            grad_outputs: vec![Some(v.clone())],
            create_graph: true,
            ..Default::default()
        },
    )
    .remove(0)
    .unwrap();
    assert_eq!((g.device(), g.dtype()), (Device::Cpu, DType::F32));
    let gv = autograd::grad(
        &[&g],
        &[&v],
        GradOptions {
            grad_outputs: vec![Some(homura::full([2], 5.0))],
            ..Default::default()
        },
    )
    .remove(0)
    .unwrap();
    assert_eq!((gv.device(), gv.dtype()), (GPU, DType::F64));
    assert_eq!(gv.to_vec::<f64>(), [5.0; 2]);
}

#[test]
fn copy_between_devices_sends_the_gradient_back() {
    let src = homura::ones([2, 3]);
    src.requires_grad_(true);
    let dst = homura::zeros_with([2, 3], (DType::F16, GPU));
    dst.copy_(&src);
    assert_eq!(dst.to_vec::<f32>(), [1.0; 6]);
    dst.backward_with(&homura::full_with([2, 3], 4.0, (DType::F16, GPU)));
    let g = src.grad().unwrap();
    assert_eq!((g.device(), g.dtype()), (Device::Cpu, DType::F32));
    assert_eq!(g.to_vec::<f32>(), [4.0; 6]);
}

/// The identity, whose backward returns a gradient of ones on the CPU.
struct CpuGrad;

impl Function for CpuGrad {
    fn forward(&mut self, _ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].deep_clone()]
    }

    fn backward(
        &self,
        _ctx: &BackwardContext<'_>,
        _grads: &[Option<Tensor>],
    ) -> Vec<Option<Tensor>> {
        vec![Some(homura::ones([]))]
    }
}

#[test]
fn a_zero_dimensional_gradient_moves_to_the_input_device() {
    let x = homura::ones_with([], GPU);
    x.requires_grad_(true);
    let y = CpuGrad.apply(&[&x]).remove(0);
    assert_eq!(y.device(), GPU);
    y.backward_with(&homura::ones_with([], GPU));
    let g = x.grad().unwrap();
    assert_eq!(g.device(), GPU);
    assert_eq!(g.item::<f32>(), 1.0);
}

#[derive(Module)]
struct Net {
    weight: Parameter,
    bias: Parameter,
    count: Tensor,
    #[module(non_persistent)]
    cache: Tensor,
}

#[test]
fn modules_move_to_the_gpu_and_back() {
    let net = Net {
        weight: Parameter::new(homura::arange_with(6, DType::F32).reshape([2, 3])),
        bias: Parameter::new(homura::zeros([2])),
        count: homura::full([], 7i64),
        cache: homura::ones_with([3], DType::F16),
    };
    let weight = net.weight.clone();
    weight.set_grad(Some(homura::full([2, 3], 0.5)));
    let grad = weight.grad().unwrap();

    net.to(GPU);
    // The parameter keeps its identity; its data, gradient and the buffers moved.
    assert!(net.weight.is_same(&weight));
    assert!(weight.is_leaf() && weight.requires_grad());
    for t in [&*net.weight, &*net.bias, &net.count, &net.cache] {
        assert_eq!(t.device(), GPU);
    }
    assert_eq!(weight.to_vec::<f32>(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(net.count.item::<i64>(), 7);
    assert_eq!(net.cache.dtype(), DType::F16);
    let moved = weight.grad().unwrap();
    assert!(!moved.is_same(&grad) && grad.device() == Device::Cpu);
    assert_eq!(moved.device(), GPU);
    assert!(!moved.requires_grad());
    assert_eq!(moved.to_vec::<f32>(), [0.5; 6]);

    // To f64 on the GPU, then back to the CPU.
    net.to(DType::F64);
    assert_eq!((weight.device(), weight.dtype()), (GPU, DType::F64));
    assert_eq!(net.count.dtype(), DType::I64);
    net.to(Device::Cpu);
    assert!(net.weight.is_same(&weight));
    for t in [&*net.weight, &*net.bias, &net.count, &net.cache] {
        assert_eq!(t.device(), Device::Cpu);
    }
    assert_eq!(weight.to_vec::<f64>(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(weight.grad().unwrap().device(), Device::Cpu);
    assert_eq!(weight.grad().unwrap().to_vec::<f64>(), [0.5; 6]);
}
